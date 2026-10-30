"""The whole recycling trunk, torch/MPS (production call) vs the MLX trunk, as the pipeline sees it.

Both include the torch prelude (input embedder, s_init, z_init) and, for MLX, the torch<->MLX
conversions, so this is the cost that actually replaces `get_pairformer_output`. NO diffusion and no
confidence head, so memory stays far below a full prediction; one process per N, hard-capped.

Torch is called the way `mol pred` calls it (inplace_safe=True, chunk_size=1024). The MSA is a
64-row synthetic alignment (the per-cycle sampler draws a random subset, identically in both).

    PYTHONPATH=. python bench/trunk_bench.py case <N> <cycles> <cap_gib>
    PYTHONPATH=. python bench/trunk_bench.py          # driver
"""
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
CKPT = "/Users/ethanputnam/Mol-1/data_cache/checkpoint/mol-1-base.pt"
MSA_ROWS = 64


def build(n_total: int, cycles: int):
    """Load the real model on MPS and featurize an (N=n_total) two-chain job. Returns
    (model, features, kwargs) ready for ``model.get_pairformer_output(input_feature_dict=f, **kw)``.
    Call inside a memory_guard."""
    import torch
    from capture_recycle_refs import BINDER, TARGET_POOL, deep_update, synthetic_a3m
    from mol.config.base import configs as configs_base
    from mol.config.data import data_configs
    from mol.config.inference import inference_configs
    from mol.config.model_type import model_configs
    from mol.config.parser import parse_configs
    from mol.data.inference.infer_dataloader import InferenceDataset
    from mol.model.mol import update_input_feature_dict
    from mol.runner.inference import InferenceRunner
    from mol.utils.torch_utils import to_device

    tmp = tempfile.mkdtemp(prefix="trunkbench_")
    target = (TARGET_POOL * (n_total // len(TARGET_POOL) + 1))[: n_total - len(BINDER)]   # same prefix as before for N<=572
    open(f"{tmp}/A.a3m", "w").write(synthetic_a3m(target, MSA_ROWS, seed=n_total))
    open(f"{tmp}/B.a3m", "w").write(f">query\n{BINDER}\n")
    job = [{"name": "tb", "sequences": [
        {"proteinChain": {"id": ["A"], "sequence": target, "count": 1, "unpairedMsaPath": f"{tmp}/A.a3m"}},
        {"proteinChain": {"id": ["B"], "sequence": BINDER, "count": 1, "unpairedMsaPath": f"{tmp}/B.a3m"}}]}]
    json.dump(job, open(f"{tmp}/in.json", "w"))
    arg_str = (f"--input_json_path {tmp}/in.json --dump_dir {tmp}/out --use_msa true --use_template false "
               f"--device mps --load_checkpoint_path {CKPT} --triangle_multiplicative auto --triangle_attention auto")
    base = {**configs_base, **{"data": data_configs}, **inference_configs}
    first = parse_configs(configs=dict(base), arg_str=arg_str, fill_required_with_null=True)
    deep_update(base, model_configs[first.model_name])
    configs = parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)
    runner = InferenceRunner(configs)
    model = runner.model.eval()
    data, _, err = InferenceDataset(configs)[0]
    assert not err, err
    feats = to_device(data["input_feature_dict"], runner.device)
    assert feats["restype"].shape[0] == n_total
    f = update_input_feature_dict(model.relative_position_encoding.generate_relp(dict(feats)))
    return model, f, dict(N_cycle=cycles, inplace_safe=True, chunk_size=1024)


def case(n_total: int, cycles: int, cap: float, arms: str = "torch,inproc,worker", worker_cap: float = None) -> None:
    """``arms``: comma list of torch / inproc / worker. Missing arms print as nan."""
    import torch
    from mol_mlx.guard import memory_guard
    from mol_mlx.hybrid import install

    arms = set(arms.split(","))
    nan = float("nan")
    with memory_guard(max_gib=cap, extra=[lambda: torch.mps.driver_allocated_memory() / 2**30]) as peak:
        model, f, kw = build(n_total, cycles)

        def timed(fn, reps=2):
            best = 1e9
            for _ in range(reps):
                torch.manual_seed(0)
                torch.mps.synchronize()
                t = time.perf_counter()
                with torch.no_grad():
                    out = fn(input_feature_dict=f, **kw)
                torch.mps.synchronize()
                best = min(best, time.perf_counter() - t)
                del out
            return best

        original = model.get_pairformer_output
        t_torch = torch_gib = nan
        if "torch" in arms:
            timed(original, reps=1)                       # warm-up (kernel compilation)
            t_torch = timed(original)
            torch_gib = peak["extra"]
        t_mlx = nan
        if "inproc" in arms:
            install(model, os.path.join(ROOT, "weights"), isolate=False)
            timed(model.get_pairformer_output, reps=1)    # warm-up (MLX graph/kernels)
            t_mlx = timed(model.get_pairformer_output)
            model.get_pairformer_output = original
        t_iso, worker_gib = nan, nan
        if "worker" in arms:
            remote = install(model, os.path.join(ROOT, "weights"), isolate=True, worker_cap_gib=worker_cap or cap)
            timed(model.get_pairformer_output, reps=1)    # warm-up (worker start, kernels)
            t_iso = timed(model.get_pairformer_output)
            worker_gib = remote.last_mlx_peak_gib         # the worker's own sampled MLX peak (RSS cannot see Metal buffers)
            remote.close()
        print(f"OK N={n_total} cycles={cycles} torch_s={t_torch:.2f} mlx_s={t_mlx:.2f} iso_s={t_iso:.2f} "
              f"speedup={t_torch / t_mlx:.2f} iso_speedup={t_torch / t_iso:.2f} torch_peak_gib={torch_gib:.1f} "
              f"max_held_gib={max(peak['extra'], peak['mlx']):.1f} worker_gib={worker_gib:.1f} msa_rows={MSA_ROWS}")


def driver(cap: float, sizes=(97, 128, 192, 256), arms: str = "torch,inproc,worker", worker_cap: float = None) -> None:
    print(f"trunk only (no diffusion), {MSA_ROWS}-row MSA, 4 cycles, arms={arms}, hard cap {cap} GiB per process\n")
    print(f"{'N':>4s} {'torch (s)':>10s} {'MLX in-proc':>12s} {'MLX worker':>11s} {'in-proc':>8s} {'worker':>8s}"
          f"  {'torch GiB':>10s} {'parent GiB':>10s} {'worker GiB':>10s}")
    for n in sizes:
        cmd = [sys.executable, __file__, "case", str(n), "4", str(cap), arms] + ([str(worker_cap)] if worker_cap else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env={**os.environ, "PYTHONPATH": ROOT})
        line = next((l for l in r.stdout.splitlines() if l.startswith("OK ")), None)
        if line:
            f = dict(kv.split("=") for kv in line.split()[1:])
            print(f"{n:4d} {float(f['torch_s']):10.2f} {float(f['mlx_s']):12.2f} {float(f['iso_s']):11.2f} "
                  f"{float(f['speedup']):7.2f}x {float(f['iso_speedup']):7.2f}x  "
                  f"{float(f['torch_peak_gib']):10.1f} {float(f['max_held_gib']):10.1f} {float(f['worker_gib']):10.1f}")
        else:
            killed = "memory_guard" in (r.stderr or "")
            tail = (r.stderr or "").strip().splitlines()[-1:] or ["no output"]
            print(f"{n:4d} {'OOM (guard)' if killed else 'ERR: ' + tail[0][:80]}")
        sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "case":
        case(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]),
             sys.argv[5] if len(sys.argv) > 5 else "torch,inproc,worker",
             float(sys.argv[6]) if len(sys.argv) > 6 else None)
    else:
        # driver [cap_gib] [sizes comma list] [arms comma list] [worker_cap_gib]
        a = sys.argv[1:]
        driver(float(a[0]) if a else 12.0,
               tuple(int(x) for x in a[1].split(",")) if len(a) > 1 else (97, 128, 192, 256),
               a[2] if len(a) > 2 else "torch,inproc,worker",
               float(a[3]) if len(a) > 3 else None)
