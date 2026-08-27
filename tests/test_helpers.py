import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.data.cache import cache_key, cached_frame
from src.utils.helpers import (
    PROJECT_ROOT,
    load_config,
    load_pickle,
    plot_metrics,
    resolve_path,
    save_pickle,
)


def test_project_root_points_at_the_repo():
    assert (PROJECT_ROOT / "main.py").exists()
    assert (PROJECT_ROOT / "src" / "config" / "config.yaml").exists()


def test_resolve_path_anchors_relative_paths():
    assert resolve_path("data/model.pkl") == PROJECT_ROOT / "data" / "model.pkl"


def test_resolve_path_leaves_absolute_paths_alone(tmp_path):
    assert resolve_path(tmp_path) == tmp_path


def test_config_loads_from_any_working_directory(tmp_path, monkeypatch):
    """The pipeline must not depend on being launched from the repo root."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert "als" in cfg and "data" in cfg


def test_config_loads_in_a_subprocess_from_another_cwd():
    """Belt and braces: a real interpreter, started somewhere else entirely."""
    result = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r});"
         "from src.utils.helpers import load_config; print(sorted(load_config()))"],
        cwd=os.path.expanduser("~"), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "als" in result.stdout


def test_config_is_read_as_utf8(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text('title: "Sony WH-1000XM5 — Headphones “Pro”"\n', encoding="utf-8")
    assert "—" in load_config(path)["title"]


def test_pickle_round_trip_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "obj.pkl"
    save_pickle({"a": 1}, target)
    assert target.exists()
    assert load_pickle(target) == {"a": 1}


def test_cache_key_is_stable_and_order_independent():
    assert cache_key("a", 1) == cache_key("a", 1)
    assert cache_key({"x": 1, "y": 2}) == cache_key({"y": 2, "x": 1})
    assert cache_key("a", 1) != cache_key("a", 2)


def test_cached_frame_hits_on_second_call(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return pd.DataFrame({"x": [1, 2, 3]})

    cached_frame("t", build, tmp_path, ["v1"])
    cached_frame("t", build, tmp_path, ["v1"])
    assert len(calls) == 1


def test_cached_frame_invalidates_when_config_changes(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return pd.DataFrame({"x": [1]})

    cached_frame("t", build, tmp_path, ["v1"])
    cached_frame("t", build, tmp_path, ["v2"])
    assert len(calls) == 2


def test_cached_frame_can_be_disabled(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return pd.DataFrame({"x": [1]})

    cached_frame("t", build, tmp_path, ["v1"], enabled=False)
    cached_frame("t", build, tmp_path, ["v1"], enabled=False)
    assert len(calls) == 2
    assert not list(Path(tmp_path).glob("*.parquet"))


def test_plot_metrics_writes_a_chart(tmp_path):
    out = plot_metrics(
        {"ALS": {"NDCG@10": 0.013}, "Popularity": {"NDCG@10": 0.0098}},
        tmp_path / "chart.png",
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_metrics_handles_empty_results(tmp_path):
    assert plot_metrics({}, tmp_path / "none.png") is None
