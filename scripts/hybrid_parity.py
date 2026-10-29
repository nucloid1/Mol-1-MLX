"""End-to-end parity: Mol-1's torch trunk vs the MLX trunk, same loaded model, same seed.

One full prediction (4 recycles, 20 diffusion steps, 2 samples) with the original torch
`get_pairformer_output`, then the same on the SAME model after `hybrid.install`. The diffusion
noise and MSA sampling draw from the CPU generator in both runs, so any difference comes from the
trunk alone. Raw (unaligned) coordinate differences are meaningful because both runs start from
the same noise. Small N only (default 97), capped.

    PYTHONPATH=/Users/ethanputnam/Mol-1-MLX python hybrid_parity.py [N]
"""
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

from mol_mlx.guard import memory_guard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_recycle_refs import BINDER, TARGET_POOL, deep_update, synthetic_a3m  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = "/Users/ethanputnam/Mol-1/data_cache/checkpoint/mol-1-base.pt"

# Route every torch.randn / randperm through the CPU generator so both runs see identical noise.
for _name in ("randn", "randperm"):
    _fn = getattr(torch, _name)

    def _on_cpu(*a, _fn=_fn, device=None, **k):
        out = _fn(*a, device="cpu", **k)
        return out if device is None else out.to(device)
    setattr(torch, _name, _on_cpu)


def main(n_total: int) -> None:
    from mol.config.base import configs as configs_base
    from mol.config.data import data_configs
    from mol.config.inference import inference_configs
    from mol.config.model_type import model_configs
    from mol.config.parser import parse_configs
    from mol.data.inference.infer_dataloader import InferenceDataset
    from mol.runner.inference import InferenceRunner
    from mol.utils.torch_utils import to_device
    from mol_mlx.hybrid import install

    tmp = tempfile.mkdtemp(prefix="parity_")
    target = (TARGET_POOL * 2)[: n_total - len(BINDER)]
    open(f"{tmp}/A.a3m", "w").write(synthetic_a3m(target, 24, seed=n_total))
    open(f"{tmp}/B.a3m", "w").write(f">query\n{BINDER}\n")
    job = [{"name": "parity", "sequences": [
        {"proteinChain": {"id": ["A"], "sequence": target, "count": 1, "unpairedMsaPath": f"{tmp}/A.a3m"}},
        {"proteinChain": {"id": ["B"], "sequence": BINDER, "count": 1, "unpairedMsaPath": f"{tmp}/B.a3m"}}]}]
    json.dump(job, open(f"{tmp}/in.json", "w"))
    arg_str = (f"--input_json_path {tmp}/in.json --dump_dir {tmp}/out --use_msa true --use_template false "
               f"--device mps --load_checkpoint_path {CKPT} --triangle_multiplicative auto "
               f"--triangle_attention auto --model.N_cycle 4 --sample_diffusion.N_step 20 "
               f"--sample_diffusion.N_sample 2")
    base = {**configs_base, **{"data": data_configs}, **inference_configs}
    first = parse_configs(configs=dict(base), arg_str=arg_str, fill_required_with_null=True)
    deep_update(base, model_configs[first.model_name])
    configs = parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)

    runner = InferenceRunner(configs)
    data, _, err = InferenceDataset(configs)[0]
    assert not err, err
    data = to_device(data, runner.device)
    print(f"N_token {int(data['N_token'])}  N_atom {int(data['N_atom'])}  device {runner.device}", flush=True)

    def run(label):
        torch.manual_seed(0)
        torch.mps.synchronize()
        t = time.perf_counter()
        pred = runner.predict({"input_feature_dict": dict(data["input_feature_dict"]), **{k: v for k, v in data.items() if k != "input_feature_dict"}})
        torch.mps.synchronize()
        dt = time.perf_counter() - t
        print(f"  {label:12s} {dt:6.1f} s", flush=True)
        return pred, dt

    print("torch trunk (original):")
    a, ta = run("torch")

    # (1) The trunk's own outputs on the live features, torch vs MLX, same RNG, production-style call.
    model = runner.model
    original = model.get_pairformer_output
    feats = dict(data["input_feature_dict"])
    prod = dict(N_cycle=4, inplace_safe=True, chunk_size=1024)
    from mol.model.mol import update_input_feature_dict
    f = update_input_feature_dict(model.relative_position_encoding.generate_relp(dict(feats)))
    torch.manual_seed(0)
    with torch.no_grad():
        _, s_t, z_t = original(input_feature_dict=f, **prod)
    s_t, z_t = s_t.float().cpu().numpy(), z_t.float().cpu().numpy()

    # (2) Control: the SAME torch trunk with its output perturbed at the size of the MLX-vs-torch
    # difference (3e-6 relative). If this moves the coordinates as much as the hybrid does, the
    # divergence is the sampler's sensitivity on a poorly determined complex, not a trunk error.
    gen = torch.Generator().manual_seed(1234)

    def perturbed(**kw):
        s_in, s, z = original(**kw)
        z = z * (1.0 + 3e-6 * torch.randn(z.shape, generator=gen).to(z.device))
        return s_in, s, z
    model.get_pairformer_output = perturbed
    print("torch trunk with a 3e-6 relative perturbation (control):")
    c, _ = run("perturbed")
    model.get_pairformer_output = original

    install(model, os.path.join(ROOT, "weights"))
    print("MLX trunk (hybrid):")
    run("warm-up")
    b, tb = run("hybrid")
    torch.manual_seed(0)
    with torch.no_grad():
        _, s_m, z_m = model.get_pairformer_output(input_feature_dict=f, **prod)
    s_m, z_m = s_m.float().cpu().numpy(), z_m.float().cpu().numpy()
    rel = lambda x, y: float(np.abs(x - y).max()) / float(np.abs(y).max())
    print(f"\ntrunk output, torch vs MLX on the live features: z rel {rel(z_m, z_t):.2e}   s rel {rel(s_m, s_t):.2e}")

    ca, cb, cc = (x["coordinate"].float().cpu().numpy() for x in (a, b, c))
    print(f"\ncoordinates [{ca.shape[0]} samples x {ca.shape[1]} atoms], raw difference (same noise, no alignment):")
    for label, other in (("torch vs hybrid  ", cb), ("torch vs perturbed", cc)):
        d = np.linalg.norm(ca - other, axis=-1)
        print(f"  {label}: RMS {np.sqrt((d ** 2).mean()):8.4f} A   max {d.max():8.4f} A   per-sample RMS {[round(float(np.sqrt((x ** 2).mean())), 3) for x in d]}")
    for key in ("plddt", "ptm", "iptm", "ranking_score", "gpde"):
        va = [float(x[key]) for x in a["summary_confidence"]]
        vb = [float(x[key]) for x in b["summary_confidence"]]
        print(f"  {key:14s} torch {np.round(va, 4)}  hybrid {np.round(vb, 4)}  max |diff| {max(abs(x - y) for x, y in zip(va, vb)):.2e}")
    print(f"\ntime: torch {ta:.1f} s, hybrid {tb:.1f} s ({ta / tb:.2f}x) at N={n_total}, 4 cycles")


if __name__ == "__main__":
    with memory_guard(max_gib=14.0, extra=[lambda: torch.mps.driver_allocated_memory() / 2**30]):
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 97)
