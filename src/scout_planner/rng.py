"""Independent, named random streams derived from one run seed (D-015).

Every random draw in the project goes through a ``numpy.random.Generator`` that
belongs to exactly one *purpose* ("arrivals", "durations", "rework", ...). This
is the *common random numbers* technique: when two runs differ in one parameter
(say, the assignment policy), each purpose still sees the same random numbers,
so differences in results come from the parameter, not from different luck.

How a stream is derived
-----------------------
``SeedSequence(seed, spawn_key=(name_key(name),))`` where ``name_key`` is a
64-bit integer taken from a BLAKE2b hash of the name. ``spawn_key`` is the
same mechanism ``SeedSequence.spawn()`` uses internally for child sequences,
so the children are statistically independent of each other and of the root.

Why key by a hash of the name instead of calling ``SeedSequence(seed).spawn(n)``:
``spawn`` numbers children 0, 1, 2, ... in request order, so stream "rework"
would get different numbers depending on whether it was requested first or
fifth, or whether a new stream was added later. Keying by name makes each
stream depend only on ``(seed, name)``.

Why not Python's built-in ``hash(name)``: it is salted per process
(``PYTHONHASHSEED``), so it changes between runs. A cryptographic digest is
stable across processes, machines and Python versions. 64 bits makes a
collision between two stream names practically impossible; we still check.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def name_key(name: str) -> int:
    """Stable 64-bit integer for a stream name (same on every machine and process)."""
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def make_stream(seed: int, name: str) -> np.random.Generator:
    """One generator for one purpose. Depends only on ``(seed, name)``."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if not name:
        raise ValueError("stream name must be a non-empty string")
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(name_key(name),)))


def make_streams(seed: int, names: Sequence[str]) -> dict[str, np.random.Generator]:
    """Create one independent generator per name, keyed by name.

    The generator for a given name is identical whatever other names are
    requested and in whatever order. For replications, call this with a
    different ``seed`` per replication (e.g. ``base_seed + k``): SeedSequence
    hashes its entropy, so adjacent integer seeds give unrelated streams.
    """
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate stream names: {sorted(names)}")
    keys = [name_key(n) for n in names]
    if len(set(keys)) != len(keys):  # pragma: no cover - 64-bit collision
        raise ValueError(f"stream name hash collision among {sorted(names)}")
    return {name: make_stream(seed, name) for name in names}
