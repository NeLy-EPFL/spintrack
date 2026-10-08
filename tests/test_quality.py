"""Run quality summary: episode detection, the terminal block and the JSON sidecar."""

import json

import numpy as np

from spintrack.quality import (
    episodes_above_baseline,
    format_summary,
    summarize_run,
    write_sidecar,
)


def make_trace(rng, n=1200, block=(400, 700), spike=(900, 905)):
    """Lognormal cost noise around 0.06 with one long block and one short spike at
    5x.
    """
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


def test_summary_and_sidecar(tmp_path):
    """Dropped frames are an episode with times; the sidecar keeps what the block
    skips.
    """
    n = 200
    frames = np.arange(n)
    ok = np.ones(n, bool)
    ok[50:70] = False
    cost = np.full(n, 0.05)
    q = summarize_run(
        frames,
        10.0 * frames,
        ok,
        cost,
        np.full(n, 3.0),
        ["map"] * n,
        np.zeros((n, 3)),
        illumination={"illum_peak": 0.3},
    )
    q.checks.update({"radius": "silhouette 100.0 px", "camera position": "from config"})
    assert (q.n_frames, q.n_tracked, q.n_dropped) == (200, 180, 20)
    assert [(e.start, e.end) for e in q.episodes] == [(50, 69)]
    block = format_summary(q)
    assert "frames 50-69, 0.5-0.7 s" in block and "radius: silhouette" in block
    assert len(block.splitlines()) <= 6
    path = write_sidecar(tmp_path / "summary.json", q, {"config": "config.toml"})
    loaded = json.loads(path.read_text())
    assert loaded["provenance"]["config"] == "config.toml"
    assert loaded["quality"]["checks"]["camera position"] == "from config"
    assert loaded["quality"]["illumination"] == {"illum_peak": 0.3}
