"""``mol1-mlx``: Mol-1's own CLI with the recycling trunk running in MLX.

    mol1-mlx pred -i job.json -o out --device mps ...      # every flag `mol pred` takes

It does not fork or copy Mol-1. It patches ONE method, ``InferenceRunner.load_checkpoint``, so that
once the torch weights are loaded the trunk is replaced by the MLX one (``hybrid.install``), then
hands over to Mol-1's click group. Everything else (featurization, diffusion, confidence, output)
is Mol-1's code.

Environment:
    MOLMLX_WEIGHTS          converted weights directory (default: <repo>/weights; build it with
                            scripts/convert_weights.py)
    MOLMLX_DISABLE=1        run plain torch (the wrapper becomes a no-op)
    MOLMLX_ISOLATE=0        run MLX inside the torch process instead of a worker (slower, see worker.py)
    MOLMLX_WORKER_CAP_GIB   kill the MLX worker if it passes this much memory
"""

from __future__ import annotations

import logging
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("mol_mlx.cli")


def _truthy(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() not in ("0", "false", "no", "off", "")


def patch_runner() -> None:
    """Make every ``InferenceRunner`` use the MLX trunk once its checkpoint is loaded."""
    from mol.runner.inference import InferenceRunner

    from .hybrid import install

    weights = os.environ.get("MOLMLX_WEIGHTS", os.path.join(ROOT, "weights"))
    original = InferenceRunner.load_checkpoint

    def load_checkpoint(self) -> None:
        original(self)
        if self.device.type == "cuda":
            logger.warning("mol1-mlx: CUDA device selected, leaving the torch trunk in place")
            return
        if not os.path.exists(os.path.join(weights, "pairformer_stack.safetensors")):
            raise FileNotFoundError(
                f"no converted MLX weights in {weights}. Build them once with:\n"
                f"    python {os.path.join(ROOT, 'scripts', 'convert_weights.py')}\n"
                f"or set MOLMLX_WEIGHTS."
            )
        cap = os.environ.get("MOLMLX_WORKER_CAP_GIB")
        install(self.model, weights, isolate=_truthy("MOLMLX_ISOLATE", True),
                worker_cap_gib=float(cap) if cap else None)

    InferenceRunner.load_checkpoint = load_checkpoint


def main() -> None:
    if not _truthy("MOLMLX_DISABLE", False):
        patch_runner()
    from mol.runner.batch_inference import mol_cli

    mol_cli(prog_name="mol1-mlx")


if __name__ == "__main__":
    main()
