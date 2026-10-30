"""The isolated MLX worker must return what the in-process Trunk returns, on the real model's inputs.

Inputs are the captured N=48 activations/features and the exact per-cycle MSA samples from
scripts/capture_recycle_refs.py. One worker process serves every test in this module.

    PYTHONPATH=. pytest tests/test_worker.py -q
"""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from mol_mlx.msa import Trunk
from mol_mlx.template import pair_features
from mol_mlx.worker import RemoteTrunk, WorkerError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights")
N = 48

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "refs", f"recycle_n{N}.npz")), reason="recycle refs not captured"
)


@pytest.fixture(scope="module")
def case():
    r = np.load(os.path.join(ROOT, "refs", f"recycle_n{N}.npz"))
    cycles = sorted({int(k.split(".")[1][5:]) for k in r.files if k.startswith("msa.cycle") and ".sample." in k})
    samples = [{k[len(f"msa.cycle{c}.sample."):]: r[k] for k in r.files if k.startswith(f"msa.cycle{c}.sample.")}
               for c in cycles]
    feat = {k[5:]: r[k] for k in r.files if k.startswith("feat.")}
    return {"s_inputs": r["s_inputs"], "s_init": r["s_init"], "z_init": r["z_init"], "samples": samples,
            "feat": feat}


@pytest.fixture(scope="module")
def worker():
    w = RemoteTrunk(WEIGHTS)
    yield w
    w.close()


def rel(got, want):
    return float(np.abs(got - want).max()) / (float(np.abs(want).max()) or 1.0)


def test_worker_matches_in_process_trunk(case, worker):
    n_cycle = len(case["samples"])
    s_w, z_w = worker(case["s_inputs"], case["s_init"], case["z_init"], n_cycle, case["samples"],
                      template_feat=case["feat"])
    trunk = Trunk(WEIGHTS)
    s_l, z_l = trunk(mx.array(case["s_inputs"]), mx.array(case["s_init"]), mx.array(case["z_init"]), n_cycle,
                     sample_fn=lambda c: case["samples"][c], template_at=pair_features(case["feat"]))
    mx.eval(s_l, z_l)
    assert s_w.shape == case["s_init"].shape and z_w.shape == case["z_init"].shape
    assert s_w.dtype == np.float32 and z_w.dtype == np.float32
    assert rel(z_w, np.array(z_l)) < 1e-5, rel(z_w, np.array(z_l))
    assert rel(s_w, np.array(s_l)) < 1e-5, rel(s_w, np.array(s_l))


def test_worker_is_reused_and_repeatable(case, worker):
    """A second request goes to the same process and gives the same answer; no shm blocks leak."""
    args = (case["s_inputs"], case["s_init"], case["z_init"], len(case["samples"]), case["samples"])
    s1, z1 = worker(*args, template_feat=case["feat"])
    pid = worker.proc.pid
    s2, z2 = worker(*args, template_feat=case["feat"])
    assert worker.proc.pid == pid
    assert rel(z2, z1) < 1e-6 and rel(s2, s1) < 1e-6
    assert worker.last_seconds and worker.last_seconds > 0


def test_worker_without_templates_runs(case, worker):
    """No template features -> the embedder is skipped (matches a feature dict with no slots)."""
    s, z = worker(case["s_inputs"], case["s_init"], case["z_init"], 1, case["samples"][:1])
    assert np.isfinite(z).all() and np.isfinite(s).all()


def test_worker_error_is_raised_not_swallowed(case, worker):
    bad = {**case["samples"][0], "msa": case["samples"][0]["msa"][:, :5]}     # wrong token count
    with pytest.raises(WorkerError, match="raised"):
        worker(case["s_inputs"], case["s_init"], case["z_init"], 1, [bad])
    # and the worker survives a failed request
    s, z = worker(case["s_inputs"], case["s_init"], case["z_init"], 1, case["samples"][:1])
    assert np.isfinite(z).all()


def test_worker_dead_between_requests_restarts(case):
    w = RemoteTrunk(WEIGHTS)
    try:
        w.start()
        old = w.proc.pid
        w.proc.kill()
        w.proc.wait()
        s, z = w(case["s_inputs"], case["s_init"], case["z_init"], 1, case["samples"][:1])
        assert w.proc.pid != old and np.isfinite(z).all()
    finally:
        w.close()


def test_worker_killed_mid_run_raises(case):
    """Killed during a request (what its memory guard or the OOM reaper would do): a loud error,
    never a silent fallback, and the next request starts a fresh worker."""
    import threading
    import time

    w = RemoteTrunk(WEIGHTS)
    try:
        w.start()
        many = [case["samples"][c % len(case["samples"])] for c in range(40)]      # long enough to catch
        threading.Thread(target=lambda: (time.sleep(0.5), w.proc.kill()), daemon=True).start()
        with pytest.raises(WorkerError, match="died"):
            w(case["s_inputs"], case["s_init"], case["z_init"], len(many), many, template_feat=case["feat"])
        s, z = w(case["s_inputs"], case["s_init"], case["z_init"], 1, case["samples"][:1])
        assert np.isfinite(z).all()
    finally:
        w.close()
