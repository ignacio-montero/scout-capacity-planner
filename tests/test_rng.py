"""Named random streams (D-015): deterministic and independent of request order."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from scout_planner.rng import make_stream, make_streams, name_key


def draws(gen: np.random.Generator) -> list[float]:
    return gen.random(5).tolist()


def test_same_seed_and_name_give_same_numbers() -> None:
    assert draws(make_stream(42, "arrivals")) == draws(make_stream(42, "arrivals"))


def test_stream_does_not_depend_on_other_names_or_their_order() -> None:
    a = make_streams(42, ["arrivals", "durations", "rework"])
    b = make_streams(42, ["rework", "arrivals"])
    c = make_streams(42, ["rework", "freelance_hours", "durations", "arrivals"])
    assert draws(a["arrivals"]) == draws(b["arrivals"]) == draws(c["arrivals"])
    assert draws(a["rework"]) == draws(b["rework"]) == draws(c["rework"])


def test_different_names_give_different_streams() -> None:
    streams = make_streams(42, ["arrivals", "durations"])
    assert draws(streams["arrivals"]) != draws(streams["durations"])


def test_different_seeds_give_different_streams() -> None:
    assert draws(make_stream(42, "arrivals")) != draws(make_stream(43, "arrivals"))


def test_drawing_from_one_stream_does_not_shift_another() -> None:
    """The core of common random numbers: extra draws elsewhere change nothing here."""
    lean = make_streams(7, ["rework", "arrivals"])
    busy = make_streams(7, ["rework", "arrivals"])
    busy["rework"].random(1000)  # e.g. one policy triggers many more rework samples
    assert draws(lean["arrivals"]) == draws(busy["arrivals"])


def test_streams_are_stable_across_processes() -> None:
    """Unlike built-in hash(), the name key ignores PYTHONHASHSEED."""
    code = "from scout_planner.rng import make_stream; print(make_stream(42, 'arrivals').random())"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for hash_seed in ("1", "2")
    }
    assert outputs == {repr(make_stream(42, "arrivals").random())}


def test_name_key_is_pinned() -> None:
    # BLAKE2b is a fixed algorithm; if this changes, every run's randomness changed.
    assert name_key("arrivals") == 1561635717968335324
    assert name_key("arrivals") != name_key("arrival")


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        make_streams(42, ["arrivals", "arrivals"])


@pytest.mark.parametrize(("seed", "name"), [(-1, "arrivals"), (42, "")])
def test_invalid_seed_or_name_is_rejected(seed: int, name: str) -> None:
    with pytest.raises(ValueError):
        make_stream(seed, name)


# --- replications (N3) -------------------------------------------------------------


def test_replication_zero_is_backward_compatible() -> None:
    """Pinned raw output: generated worlds committed before replications existed
    must not change. If this fails, every seed-0 world changed."""
    assert make_stream(42, "arrivals").bit_generator.random_raw() == 15175625775097614054
    assert draws(make_stream(42, "arrivals", replication=0)) == draws(make_stream(42, "arrivals"))


def test_replications_differ_from_each_other() -> None:
    reps = [draws(make_stream(42, "arrivals", replication=k)) for k in range(5)]
    assert len({tuple(r) for r in reps}) == 5


def test_replications_never_coincide_with_a_neighbouring_seed() -> None:
    """The bug ``seed + k`` would have: rep 1 of seed 42 == rep 0 of seed 43."""
    worlds = {
        (seed, k): tuple(draws(make_stream(seed, "arrivals", replication=k)))
        for seed in range(40, 46)
        for k in range(4)
    }
    assert len(set(worlds.values())) == len(worlds)


def test_make_streams_passes_replication_through() -> None:
    streams = make_streams(42, ["arrivals", "rework"], replication=2)
    assert draws(streams["rework"]) == draws(make_stream(42, "rework", replication=2))


def test_negative_replication_is_rejected() -> None:
    with pytest.raises(ValueError, match="replication"):
        make_stream(42, "arrivals", replication=-1)
