"""The MLX trunk in its own process.

Why a separate process: MLX runs the 48-block stack slower once torch has done its CPU-side work in
the SAME process (loading the checkpoint, featurizing). Measured on the real model at N=97, same
stack, same inputs, hard-synced timing:

    clean MLX process                          568 ms
    + torch.load of the checkpoint (CPU only)  676 ms     (cpu/wall 0.19 -> 0.35)
    checkpoint freed again                     590 ms
    full model + features built in-process     ~760 ms    (+33%)
    separate MLX process while ANOTHER process
      holds the full torch model on MPS        562 ms

Whole-trunk effect (4 cycles, in-process MLX time / worker time): 1.47x at N=97, 1.22x at 128, 1.05x at
192, 1.10x at 256, so it matters most at small N. Torch's own timings are noisy (28% run to run at
N=192); the MLX ones are stable.

Not the GPU context, resident MPS memory, Python GC (zero collections during the calls, and freeze or
disable changes nothing), torch's thread count, MLX's ops-per-buffer, a fresh thread or stream: each
was tried and made no difference. The cause is unidentified; the cure is the process boundary.

Protocol, over the worker's stdin/stdout (stray prints are sent to stderr): each message is an 8-byte
big-endian length followed by a pickle. Array payloads never travel over the pipe; the parent puts
them in a shared-memory block and passes the name and a layout table.

    parent -> worker  ("run", {in_shm, in_layout, out_shm, out_layout, n_cycle, samples, chunks})
                      ("quit", None)
    worker -> parent  ("ready", info) once, then ("ok", {"seconds": t}) or ("error", traceback)
"""

from __future__ import annotations

import argparse
import atexit
import os
import pickle
import struct
import subprocess
import sys
import traceback
from multiprocessing import resource_tracker, shared_memory
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Layout = List[Tuple[str, Tuple[int, ...], str, int]]      # (key, shape, dtype, byte offset)
_ALIGN = 64


def _send(stream, obj: Any) -> None:
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">Q", len(blob)))
    stream.write(blob)
    stream.flush()


def _recv(stream) -> Any:
    head = stream.read(8)
    if len(head) < 8:
        raise EOFError("peer closed the pipe")
    (n,) = struct.unpack(">Q", head)
    blob = stream.read(n)
    if len(blob) < n:
        raise EOFError("peer closed the pipe mid-message")
    return pickle.loads(blob)


def pack(arrays: Dict[str, np.ndarray]) -> Tuple[shared_memory.SharedMemory, Layout]:
    """Copy ``arrays`` into one new shared-memory block. Caller owns the block (close + unlink)."""
    layout: Layout = []
    offset = 0
    for key, a in arrays.items():
        a = np.ascontiguousarray(a)
        layout.append((key, tuple(a.shape), a.dtype.str, offset))
        offset += (a.nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    shm = shared_memory.SharedMemory(create=True, size=max(offset, _ALIGN))
    for (key, shape, dtype, off), a in zip(layout, arrays.values()):
        np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=off)[...] = a
    return shm, layout


def allocate(spec: Dict[str, Tuple[Tuple[int, ...], str]]) -> Tuple[shared_memory.SharedMemory, Layout]:
    """A new shared-memory block laid out for arrays of the given (shape, dtype), uninitialised."""
    layout: Layout = []
    offset = 0
    for key, (shape, dtype) in spec.items():
        layout.append((key, tuple(shape), np.dtype(dtype).str, offset))
        offset += (int(np.prod(shape)) * np.dtype(dtype).itemsize + _ALIGN - 1) // _ALIGN * _ALIGN
    return shared_memory.SharedMemory(create=True, size=max(offset, _ALIGN)), layout


def views(shm: shared_memory.SharedMemory, layout: Layout) -> Dict[str, np.ndarray]:
    return {k: np.ndarray(shape, dtype=dt, buffer=shm.buf, offset=off) for k, shape, dt, off in layout}


def _attach(name: str) -> shared_memory.SharedMemory:
    """Attach to a block the PARENT owns. Python 3.12 registers every attach with this process's
    resource tracker, which would unlink the block (and warn) when the worker exits; undo that."""
    shm = shared_memory.SharedMemory(name=name)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    return shm


# --------------------------------------------------------------------------------- worker side
def _serve(weights_dir: str, use_metal: bool, cap_gib: Optional[float]) -> None:
    import contextlib
    import time

    import mlx.core as mx

    from .guard import memory_guard
    from .msa import Trunk
    from .template import pair_features

    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                       # anything printed from here on goes to stderr
    proto_in = sys.stdin.buffer

    guard = memory_guard(max_gib=cap_gib, cache_gib=2.0) if cap_gib else contextlib.nullcontext()    # nullcontext yields None
    with guard as peak:
        trunk = Trunk(weights_dir, use_metal=use_metal)
        _send(proto_out, ("ready", {"pid": os.getpid()}))
        while True:
            try:
                kind, req = _recv(proto_in)
            except EOFError:
                return
            if kind == "quit":
                return
            try:
                t0 = time.perf_counter()
                shm_in, shm_out = _attach(req["in_shm"]), _attach(req["out_shm"])
                try:
                    a, out = views(shm_in, req["in_layout"]), views(shm_out, req["out_layout"])
                    template_at = None
                    tkeys = req.get("template_keys") or ()
                    if tkeys:
                        template_at = pair_features({k: a[k] for k in tkeys})
                    samples = req["samples"]
                    s, z = trunk(mx.array(a["s_inputs"]), mx.array(a["s_init"]), mx.array(a["z_init"]),
                                 int(req["n_cycle"]), sample_fn=lambda c: samples[c],
                                 template_at=template_at, **req["chunks"])
                    mx.eval(s, z)
                    out["s"][...] = np.asarray(s)
                    out["z"][...] = np.asarray(z)
                    del s, z, template_at, a, out
                finally:
                    shm_in.close()
                    shm_out.close()
                mx.clear_cache()
                # peak: the guard's sampled MLX active+cache maximum since the worker started (a proxy,
                # polled every 0.1 s); None when the worker runs without a cap, so no guard thread
                seconds = time.perf_counter() - t0
                if peak is not None:
                    sys.stderr.write(f"[mol_mlx.worker] trunk {seconds:.1f} s, MLX peak so far {peak['mlx']:.1f} GiB\n")
                _send(proto_out, ("ok", {"seconds": seconds,
                                         "mlx_peak_gib": None if peak is None else peak["mlx"]}))
            except Exception:
                _send(proto_out, ("error", traceback.format_exc()))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="MLX trunk worker (spawned by mol_mlx.worker.RemoteTrunk)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--no-metal", action="store_true")
    ap.add_argument("--cap-gib", type=float, default=None,
                    help="kill this worker if MLX or resident memory passes this many GiB")
    args = ap.parse_args(argv)
    _serve(args.weights, not args.no_metal, args.cap_gib)


# --------------------------------------------------------------------------------- parent side
class WorkerError(RuntimeError):
    pass


class RemoteTrunk:
    """Runs ``Trunk`` in a persistent child process. Same inputs as ``Trunk.__call__`` but numpy in,
    numpy out, and the MSA samples are drawn up front (the parent owns the RNG stream).

    The child starts on first use and lives until the parent exits. A child that dies (killed by its
    memory guard, OOM, crash) raises ``WorkerError``; this never falls back silently.
    """

    def __init__(self, weights_dir: str, use_metal: bool = True, cap_gib: Optional[float] = None) -> None:
        self.weights_dir, self.use_metal, self.cap_gib = weights_dir, use_metal, cap_gib
        self.proc: Optional[subprocess.Popen] = None
        self.last_seconds: Optional[float] = None
        self.last_mlx_peak_gib: Optional[float] = None   # worker's sampled MLX peak; needs cap_gib

    # -- lifecycle
    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        cmd = [sys.executable, "-m", "mol_mlx.worker", "--weights", self.weights_dir]
        if not self.use_metal:
            cmd.append("--no-metal")
        if self.cap_gib:
            cmd += ["--cap-gib", str(self.cap_gib)]
        env = dict(os.environ)
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        atexit.register(self.close)
        try:
            kind, _ = _recv(self.proc.stdout)
        except EOFError:
            raise WorkerError(f"MLX worker exited during startup (code {self.proc.poll()}); see stderr above")
        if kind != "ready":
            raise WorkerError(f"unexpected first message from worker: {kind!r}")

    def close(self) -> None:
        p, self.proc = self.proc, None
        if p is None or p.poll() is not None:
            return
        try:
            _send(p.stdin, ("quit", None))
            p.wait(timeout=10)
        except Exception:
            p.kill()

    # -- the call
    def __call__(self, s_inputs: np.ndarray, s_init: np.ndarray, z_init: np.ndarray, n_cycle: int,
                 samples: Sequence[Optional[dict]], template_feat: Optional[Dict[str, np.ndarray]] = None,
                 **chunks) -> Tuple[np.ndarray, np.ndarray]:
        """``samples[c]``: the cycle-c MSA sample dict (numpy) or None. ``template_feat``: raw template
        arrays for ``template.pair_features`` (None skips the embedder). Returns (s, z) as numpy."""
        self.start()
        payload = {"s_inputs": s_inputs, "s_init": s_init, "z_init": z_init}
        payload.update(template_feat or {})
        shm_in, in_layout = pack(payload)
        shm_out, out_layout = allocate({"s": (s_init.shape, "<f4"), "z": (z_init.shape, "<f4")})
        try:
            _send(self.proc.stdin, ("run", {
                "in_shm": shm_in.name, "in_layout": in_layout, "out_shm": shm_out.name,
                "out_layout": out_layout, "n_cycle": int(n_cycle), "samples": list(samples),
                "template_keys": list(template_feat or ()), "chunks": chunks,
            }))
            try:
                kind, info = _recv(self.proc.stdout)
            except (EOFError, BrokenPipeError):
                code = self.proc.poll()
                self.proc = None
                raise WorkerError(f"MLX worker died during a run (exit code {code}); see its stderr above")
            if kind == "error":
                raise WorkerError(f"MLX worker raised:\n{info}")
            self.last_seconds = info["seconds"]
            self.last_mlx_peak_gib = info.get("mlx_peak_gib")
            out = views(shm_out, out_layout)
            result = (out["s"].copy(), out["z"].copy())
            del out                       # release the buffer exports, or shm.close() raises BufferError
            return result
        finally:
            for shm in (shm_in, shm_out):
                shm.close()
                shm.unlink()


if __name__ == "__main__":
    main()
