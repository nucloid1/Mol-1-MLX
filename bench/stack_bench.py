"""The 48-block pairformer stack: MLX (Metal kernel / dense) vs torch on MPS.

One fresh process per (variant, N). Random s/z of the real shapes (timing does not
depend on values). Every process runs under a hard memory cap that also watches
torch's MPS allocator, which neither ``ps`` nor MLX's counters can see.

The memory column is the watchdog's SAMPLED maximum of memory held (it includes the
allocator's cache), polled every 0.1 s. It is a proxy, not an exact peak: MLX's own
peak counter proved unreliable at small sizes in this project, and a polled maximum can
miss a spike shorter than the interval. Timing is wall clock with a hard sync.

    stack_bench.py case <variant> <N> <cap_gib>
    stack_bench.py [cap_gib]
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = ("mlx-metal", "mlx-dense", "torch-mps", "torch-prod")  # mlx-metal uses MOLMLX_TRI_IMPL (default v2)
CKPT = "/Users/ethanputnam/Mol-1/data_cache/checkpoint/mol-1-base.pt"


def case_mlx(variant: str, N: int, cap: float) -> None:
    CHUNK = int(os.environ.get("MOLMLX_TRANSITION_CHUNK", "0")) or None   # rows per transition chunk
    import mlx.core as mx
    from mol_mlx.blocks import PairformerStack
    from mol_mlx.guard import memory_guard

    from mol_mlx import precision
    dt = {"": None, "fp32": None, "bf16": mx.bfloat16, "fp16": mx.float16}[os.environ.get("MOLMLX_PRECISION", "")]
    with memory_guard(max_gib=cap) as peak, precision.compute_dtype(dt):
        w = mx.load(os.path.join(ROOT, "weights", "pairformer_stack.safetensors"))
        stack = PairformerStack(w, use_metal=(variant == "mlx-metal"))
        s = mx.random.normal((N, 384), key=mx.random.key(0))
        z = mx.random.normal((N, N, 448), key=mx.random.key(1)) * 0.1
        mx.eval(s, z)
        t = time.perf_counter()
        mx.eval(*stack(s, z, transition_chunk=CHUNK))
        cold = time.perf_counter() - t
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            mx.eval(*stack(s, z, transition_chunk=CHUNK))
            ts.append(time.perf_counter() - t)
        print(f"OK {variant}[{os.environ.get('MOLMLX_PRECISION') or 'fp32'}] N={N} cold_s={cold:.2f} warm_s={min(ts):.2f} held_gib={peak['mlx']:.2f}")


def case_torch(N: int, cap: float, prod: bool = False) -> None:
    """``prod`` calls the stack exactly as `mol pred` does (eval + no_grad => inplace_safe=True,
    chunk_size=1024, transition chunks 128; mol/model/mol.py:991-992 and the config defaults).
    Without it: inplace_safe=False, no chunking -- the first, non-production baseline."""
    import torch
    from mol_mlx.guard import memory_guard
    from mol.config.base import configs as configs_base
    from mol.config.data import data_configs
    from mol.config.inference import inference_configs
    from mol.config.model_type import model_configs
    from mol.config.parser import parse_configs
    from mol.model.mol import Mol
    from typing import Mapping

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    mps = lambda: torch.mps.driver_allocated_memory() / 2**30
    with memory_guard(max_gib=cap, extra=[mps]) as peak:
        arg_str = (f"--input_json_path /dev/null --dump_dir /tmp/sb --device mps "
                   f"--load_checkpoint_path {CKPT} --triangle_multiplicative auto --triangle_attention auto")
        base = {**configs_base, **{"data": data_configs}, **inference_configs}
        first = parse_configs(configs=dict(base), arg_str=arg_str, fill_required_with_null=True)
        deep_update(base, model_configs[first.model_name])
        configs = parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)
        # Only the trunk stack is needed: build the model, keep that module, drop the rest.
        model = Mol(configs)
        state = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
        state = {k[len("module."):]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        del state
        stack = model.pairformer_stack.to("mps").eval()
        del model

        s = torch.randn(N, 384, device="mps")
        z = torch.randn(N, N, 448, device="mps") * 0.1
        kw = dict(pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch",
                  inplace_safe=False, chunk_size=None)
        if prod:
            kw.update(inplace_safe=True, chunk_size=1024,
                      pairmixer_transition_chunk_size=128, pairformer_transition_chunk_size=128)

        def run():
            with torch.no_grad():
                out = stack(s, z, **kw)
            torch.mps.synchronize()
            return out

        t = time.perf_counter()
        run()
        cold = time.perf_counter() - t
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            run()
            ts.append(time.perf_counter() - t)
        print(f"OK {'torch-prod' if prod else 'torch-mps'} N={N} cold_s={cold:.2f} warm_s={min(ts):.2f} held_gib={peak['extra']:.2f}")


def driver(cap: float) -> None:
    sizes = [128, 192, 256]
    print(f"hard cap {cap} GiB per process\n")
    print(f"{'N':>4s} {'variant':10s} {'cold s':>8s} {'warm s':>8s} {'held GiB':>9s}   vs torch-mps")
    for N in sizes:
        warm = {}
        for variant in VARIANTS:
            r = subprocess.run([sys.executable, __file__, "case", variant, str(N), str(cap)],
                               capture_output=True, text=True, timeout=1800,
                               env={**os.environ, "PYTHONPATH": ROOT})
            line = next((l for l in r.stdout.splitlines() if l.startswith("OK ")), None)
            if line:
                f = dict(kv.split("=") for kv in line.split()[2:])
                warm[variant] = float(f["warm_s"])
                ratio = ""
                print(f"{N:4d} {variant:10s} {float(f['cold_s']):8.2f} {warm[variant]:8.2f} {float(f['held_gib']):9.2f}   {ratio}")
            else:
                killed = "memory_guard" in (r.stderr or "")
                tail = (r.stderr or "").strip().splitlines()[-1:] or ["no output"]
                print(f"{N:4d} {variant:10s} {'OOM (guard)' if killed else 'ERR: ' + tail[0][:70]}")
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "case":
        v, n, c = sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
        case_torch(n, c, prod=(v == "torch-prod")) if v.startswith("torch") else case_mlx(v, n, c)
    else:
        driver(float(sys.argv[1]) if len(sys.argv) > 1 else 10.0)
