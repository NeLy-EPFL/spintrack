"""Run quality summary: episode detection, `.dat` summaries and the JSON sidecar."""

import json

import numpy as np

from spintrack.io.dat import N_COLUMNS, DatWriter
from spintrack.quality import (
    episodes_above_baseline,
    format_summary,
    summary_from_dat,
    write_sidecar,
)


def make_trace(rng, n=1200, block=(400, 700), spike=(900, 905)):
    """Lognormal cost noise around 0.06 with one long block and one short spike at 5x."""
    cost = 0.06 * np.exp(rng.normal(0.0, 0.25, n))
    iters = rng.integers(3, 5, n).astype(float)
    for lo, hi in (block, spike):
        cost[lo:hi] *= 5.0
        iters[lo:hi] = rng.integers(9, 12, hi - lo)
    return cost, iters, np.ones(n, bool)


def test_one_episode_per_block_and_short_spikes_ignored():
    cost, iters, ok = make_trace(np.random.default_rng(0))
    episodes = episodes_above_baseline(cost, ok, iters)
    assert len(episodes) == 1, episodes
    start, end = episodes[0]
    assert abs(start - 400) <= 25 and abs(end - 699) <= 25, episodes


def test_no_episode_without_a_block():
    rng = np.random.default_rng(1)
    cost = 0.06 * np.exp(rng.normal(0.0, 0.25, 1200))
    iters = rng.integers(3, 5, 1200).astype(float)
    assert episodes_above_baseline(cost, np.ones(1200, bool), iters) == []


def test_elevated_cost_alone_is_not_an_episode():
    """The solver has to agree: a costly but easy stretch is walking, not a failure."""
    cost, iters, ok = make_trace(np.random.default_rng(2))
    iters[:] = 3.0
    assert episodes_above_baseline(cost, ok, iters) == []
    assert episodes_above_baseline(cost, ok, None) != []


def test_dropped_frames_are_an_episode():
    rng = np.random.default_rng(3)
    cost = 0.06 * np.exp(rng.normal(0.0, 0.25, 600))
    iters = rng.integers(3, 5, 600).astype(float)
    ok = np.ones(600, bool)
    ok[200:260] = False
    assert episodes_above_baseline(cost, ok, iters) == [(200, 259)]


def _write_dat(path, frames, cost, ts):
    with DatWriter(path) as w:
        for frame, c, t in zip(frames, cost, ts, strict=True):
            values = np.zeros(N_COLUMNS)
            values[0], values[4], values[21] = frame, c, t
            values[22] = frame
            w.write(values)


def test_summary_from_dat_counts_frame_gaps_as_dropped(tmp_path):
    frames = [f for f in range(200) if not 50 <= f < 70]
    cost = np.full(len(frames), 0.05)
    _write_dat(tmp_path / "a.dat", frames, cost, [10.0 * f for f in frames])
    q = summary_from_dat(tmp_path / "a.dat")
    assert (q.n_frames, q.n_tracked, q.n_dropped) == (200, 180, 20)
    assert q.iters_median is None and q.map_coverage is None
    # The gap is the only episode, and its timestamps come from column 21.
    assert [(e.start, e.end) for e in q.episodes] == [(50, 69)]
    assert np.isclose(q.episodes[0].t0_s, 0.5)


def test_summary_from_dat_falls_back_to_fps_without_timestamps(tmp_path):
    frames = [f for f in range(200) if not 50 <= f < 70]
    cost = np.full(len(frames), 0.05)
    _write_dat(tmp_path / "b.dat", frames, cost, [-1.0] * len(frames))
    assert np.isnan(summary_from_dat(tmp_path / "b.dat").episodes[0].t0_s)
    q = summary_from_dat(tmp_path / "b.dat", fps=50.0)
    assert np.isclose(q.episodes[0].t0_s, 1.0)
    assert "1.0-1.4 s" in format_summary(q)


def test_sidecar_round_trip(tmp_path):
    frames = list(range(100))
    _write_dat(
        tmp_path / "c.dat", frames, np.full(100, 0.05), [10.0 * f for f in frames]
    )
    q = summary_from_dat(tmp_path / "c.dat")
    path = write_sidecar(tmp_path / "c-summary.json", q, {"config": "x.txt"})
    loaded = json.loads(path.read_text())
    assert loaded["provenance"]["config"] == "x.txt"
    assert loaded["quality"]["n_frames"] == 100
    assert loaded["quality"]["cost_median"] == q.cost_median
