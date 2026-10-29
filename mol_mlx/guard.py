"""Process-level memory watchdog.

``mx.set_memory_limit`` is advisory: MLX only raises when an allocation cannot be
satisfied at all, which on a 48 GB machine is long after the OS has started
swapping and the memory reaper has started killing the session. This kills the
process instead, from a background thread, as soon as either counter passes the cap.

Two counters, because neither is complete alone:

- MLX's own active + cache bytes, exact for everything MLX allocated;
- the process's resident set from ``ps``, which also sees torch/MPS and numpy.

Usage::

    from mol_mlx.guard import memory_guard
    with memory_guard(max_gib=10):
        ...
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading

import mlx.core as mx

_GIB = 1 << 30


def _rss_gib() -> float:
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return int(out) / (1 << 20)  # ps reports KiB
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def mlx_gib() -> float:
    """Bytes MLX holds right now (live arrays plus its allocator cache), in GiB."""
    return (mx.get_active_memory() + mx.get_cache_memory()) / _GIB


class MemoryExceeded(RuntimeError):
    pass


@contextlib.contextmanager
def memory_guard(max_gib: float = 10.0, interval_s: float = 0.1, cache_gib: float = 1.0,
                 extra=()):
    """Kill the process if MLX, torch/MPS or total resident memory passes ``max_gib``.

    ``extra`` is an iterable of zero-argument callables returning GiB, e.g.
    ``lambda: torch.mps.driver_allocated_memory() / 2**30`` -- torch's MPS buffers are
    invisible to both ``ps`` and MLX's counters.

    Also sets MLX's own limit slightly under the cap and bounds its allocator cache,
    so the common case is an exception rather than a kill.
    """
    mx.set_memory_limit(int(max_gib * 0.9 * _GIB))
    mx.set_cache_limit(int(cache_gib * _GIB))
    stop = threading.Event()
    peak = {"mlx": 0.0, "rss": 0.0, "extra": 0.0}
    extra = tuple(extra)

    def watch() -> None:
        while not stop.wait(interval_s):
            m, r = mlx_gib(), _rss_gib()
            e = max((fn() for fn in extra), default=0.0)
            peak["mlx"], peak["rss"], peak["extra"] = max(peak["mlx"], m), max(peak["rss"], r), max(peak["extra"], e)
            if m > max_gib or r > max_gib or e > max_gib:
                sys.stderr.write(
                    f"\n[memory_guard] KILLED: mlx {m:.2f} GiB, rss {r:.2f} GiB, extra {e:.2f} GiB"
                    f" > cap {max_gib} GiB\n"
                )
                sys.stderr.flush()
                os.kill(os.getpid(), signal.SIGKILL)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield peak
    finally:
        stop.set()
        thread.join(timeout=1)
