from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from serein.eval.split_plan import plan_splits, write_plan
from serein.eval.split_plan_plots import plot_plan


def fixture_plan():
    plan = {role: [{"start": day, "end": day}] for role, day in [
        ("train", "2021-06-01"), ("tune", "2023-05-01"),
        ("calibrate", "2023-07-01"), ("score", "2023-09-01")]}
    frame = pd.DataFrame({"issue_date": pd.to_datetime(["2021-06-01", "2023-07-01"]),
                          "point_id": ["001", "001"], "y_L0": [1., 2.]})
    return plan, frame


def test_plot_shows_absent_declared_years_and_empty_roles(tmp_path, monkeypatch):
    plan, frame = fixture_plan()
    plan["train"][0]["end"] = "2022-06-01"
    result = plan_splits({"archive": frame}, plan, leads=[0])
    tables = [result[name].copy(deep=True) for name in ["ranges", "assignments", "availability"]]
    captured = []
    real_save = matplotlib.figure.Figure.savefig

    def save(fig, *args, **kwargs):
        captured.append(fig)
        return real_save(fig, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", save)
    target = tmp_path / "plan.png"
    plot_plan(*tables, target)
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert target.stat().st_size > 10000
    fig = captured[0]
    assert [ax.get_title() for ax in fig.axes[:3]] == ["2021", "2022", "2023"]
    bars = fig.axes[-1].patches
    assert [bar.get_height() for bar in bars] == [100., 0., 100., 0.]
    assert sum(len(ax.patches) for ax in fig.axes[:3]) == 5
    assert plt.get_fignums() == []
    for name, table in zip(["ranges", "assignments", "availability"], tables):
        pd.testing.assert_frame_equal(result[name], table)


def test_plot_does_not_replace_existing_file(tmp_path):
    plan, frame = fixture_plan()
    result = plan_splits({"archive": frame}, plan, leads=[0])
    target = tmp_path / "existing.png"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        plot_plan(result["ranges"], result["assignments"], result["availability"], target)
    assert target.read_bytes() == b"keep"


def test_plot_closes_figure_on_write_failure(tmp_path, monkeypatch):
    plan, frame = fixture_plan()
    result = plan_splits({"archive": frame}, plan, leads=[0])

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail)
    with pytest.raises(OSError, match="disk failure"):
        plot_plan(result["ranges"], result["assignments"], result["availability"], tmp_path / "x.png")
    assert plt.get_fignums() == []


def test_writer_can_plot_empty_roles_and_refuses_nonboolean_flag(tmp_path):
    plan, frame = fixture_plan()
    train = tmp_path / "train.parquet"
    calibrate = tmp_path / "calibrate.parquet"
    frame.iloc[:1].to_parquet(train)
    frame.iloc[1:].to_parquet(calibrate)
    spec = tmp_path / "plan.json"
    spec.write_text(json.dumps(plan))
    output = tmp_path / "out"
    with pytest.raises(TypeError, match="plot must be boolean"):
        write_plan(train, calibrate, spec, output, leads=[0], plot="yes")
    assert not output.exists()
    write_plan(train, calibrate, spec, output, leads=[0], plot=True)
    assert (output / "development_plan.png").exists()


def test_command_exports_png_with_interactive_backend_selected(tmp_path):
    plan, frame = fixture_plan()
    train, calibrate, spec = [tmp_path / name for name in ["train.parquet", "cal.parquet", "p.json"]]
    frame.iloc[:1].to_parquet(train)
    frame.iloc[1:].to_parquet(calibrate)
    spec.write_text(json.dumps(plan))
    output = tmp_path / "cli"
    env = os.environ.copy()
    env["MPLBACKEND"] = "TkAgg"
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1] / "src"),
                                         env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-m", "serein.eval.split_plan",
                             "--train-table", str(train), "--calibrate-table", str(calibrate),
                             "--plan", str(spec), "--out", str(output), "--leads", "0", "--plot"],
                            check=False, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["calendar_separated"]
    assert (output / "development_plan.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
