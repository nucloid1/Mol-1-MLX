"""MSA module, outer product mean, and the recycling loop against the real model's activations.

References come from scripts/capture_recycle_refs.py: the model's own `get_pairformer_output` on
torch CPU, three cycles, a 24-row synthetic MSA. The exact sampled MSA of every cycle is replayed.

    PYTHONPATH=. pytest tests/test_recycle_vs_refs.py -q -s
"""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from mol_mlx.blocks import DistogramHead
from mol_mlx.msa import MSAModule, Trunk
from mol_mlx.template import TemplateEmbedder, pair_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights")
BLOCK_RTOL = 2e-5
LOOP_RTOL = 2e-4

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "refs", "recycle_n48.npz")), reason="recycle refs not captured"
)


def refs(n):
    return np.load(os.path.join(ROOT, "refs", f"recycle_n{n}.npz"))


def rel(got, want):
    g, w = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert g.shape == w.shape, f"{g.shape} != {w.shape}"
    return float(np.abs(g - w).max()) / (float(np.abs(w).max()) or 1.0)


def sample(r, c):
    p = f"msa.cycle{c}.sample."
    return {k[len(p):]: r[k] for k in r.files if k.startswith(p)}


@pytest.fixture(scope="module")
def msa_module():
    return MSAModule(mx.load(os.path.join(WEIGHTS, "msa_module.safetensors")))


@pytest.mark.parametrize("cycle", [0, 1, 2])
@pytest.mark.parametrize("b", [0, 1, 2, 3])
def test_msa_block_n48(msa_module, b, cycle):
    r = refs(48)
    p = f"msablock{b}.cycle{cycle}"
    m_out, z_out = msa_module.blocks[b](mx.array(r[f"{p}.m_in"]), mx.array(r[f"{p}.z_in"]))
    mx.eval(z_out)
    ez = rel(z_out, r[f"{p}.z_out"])
    msg = f"block {b} cycle {cycle}: z rel {ez:.2e}"
    if f"{p}.m_out" in r.files:
        em = rel(m_out, r[f"{p}.m_out"])
        msg += f", m rel {em:.2e}"
        assert em < BLOCK_RTOL, msg
    else:
        assert m_out is None, "last block must not return m"
    print("\n" + msg)
    assert ez < BLOCK_RTOL, msg


@pytest.mark.parametrize("n", [48, 97])
@pytest.mark.parametrize("cycle", [0, 1, 2])
def test_msa_module_per_cycle(msa_module, n, cycle):
    r = refs(n)
    z = msa_module(sample(r, cycle), mx.array(r[f"msa.cycle{cycle}.z_in"]), mx.array(r[f"msa.cycle{cycle}.s_inputs"]))
    mx.eval(z)
    e = rel(z, r[f"msa.cycle{cycle}.z_out"])
    print(f"\nN={n} cycle {cycle} ({sample(r, cycle)['msa'].shape[0]} rows): z rel {e:.2e}")
    assert e < BLOCK_RTOL


def features(r):
    return {k[len("feat."):]: r[k] for k in r.files if k.startswith("feat.")}


@pytest.mark.parametrize("n", [48, 97])
@pytest.mark.parametrize("cycle", [0, 1, 2])
def test_template_embedder_with_empty_slots(n, cycle):
    """Four all-gap slots, as the featurizer emits with templates off. The real output is ~29 in magnitude."""
    r = refs(n)
    emb = TemplateEmbedder(mx.load(os.path.join(WEIGHTS, "template_embedder.safetensors")))
    at = pair_features(features(r))
    assert at.shape[0] == 4
    u = emb(mx.array(r[f"tmpl.cycle{cycle}.z_in"]), at)
    mx.eval(u)
    want = r[f"tmpl.cycle{cycle}.out"]
    assert float(np.abs(want).max()) > 1.0, "reference is numerically small; this test would be vacuous"
    e = rel(u, want)
    print(f"\nN={n} cycle {cycle}: template embedder rel {e:.2e}  (|out|max {float(np.abs(want).max()):.1f})")
    assert e < BLOCK_RTOL


def test_template_slots_are_not_skippable():
    """Guard against the assumption that cost me an hour: skipping the embedder is NOT close."""
    r = refs(48)
    skipped = rel(r["msa.cycle0.z_in"] - 0.0, r["msa.cycle0.z_in"] - r["tmpl.cycle0.out"])
    print(f"\nz_in with vs without the embedder's contribution differ by rel {skipped:.2f}")
    assert skipped > 0.1


def test_chunking_matches_unchunked(msa_module):
    """Row and output-row chunking are memory knobs: they must not change the result."""
    r = refs(97)
    args = (sample(r, 1), mx.array(r["msa.cycle1.z_in"]), mx.array(r["msa.cycle1.s_inputs"]))
    full = msa_module(*args)
    chunked = msa_module(*args, msa_chunk=3, opm_chunk=16, transition_chunk=8)
    mx.eval(full, chunked)
    e = rel(chunked, np.asarray(full))
    print(f"\nchunked vs unchunked: rel {e:.2e}")
    assert e < BLOCK_RTOL


@pytest.mark.parametrize("n", [48, 97])
def test_full_recycle_loop(n):
    r = refs(n)
    trunk = Trunk(WEIGHTS)
    n_cycle = int(r["n_cycle"])
    s, z = trunk(mx.array(r["s_inputs"]), mx.array(r["s_init"]), mx.array(r["z_init"]), n_cycle,
                 sample_fn=lambda c: sample(r, c), template_at=pair_features(features(r)))
    mx.eval(s, z)
    es, ez = rel(s, r["final.s"]), rel(z, r["final.z"])
    head = DistogramHead(mx.load(os.path.join(WEIGHTS, "distogram_head.safetensors")))
    logits = head(z)
    mx.eval(logits)
    ed = rel(logits, r["final.distogram"])
    flips = float(np.mean(np.argmax(np.asarray(logits), -1) != np.argmax(r["final.distogram"], -1)))
    print(f"\nN={n}, {n_cycle} cycles: s rel {es:.2e}  z rel {ez:.2e}  distogram rel {ed:.2e}  bin flips {flips * 100:.2f}%")
    assert es < LOOP_RTOL and ez < LOOP_RTOL and ed < LOOP_RTOL
    assert flips == 0.0
