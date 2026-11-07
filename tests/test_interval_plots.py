import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from serein.eval.interval_plots import plot_coverage
from serein.eval.intervals import analyze_intervals, main


def predictions():
    return pd.DataFrame({
        "issue_date": pd.date_range("2023-06-01", periods=42), "point_id": "A",
        "y_L1": [10., 30.] * 21, "smokecast_lo_L1": 10., "smokecast_hi_L1": 20.,
        "hms_prev": [0., 2.] * 21,
    })


@pytest.mark.parametrize("groups", [[], ["point", "smoke"], ["month"]])
def test_plot_writes_png_without_mutating_summary_or_leaking_figures(tmp_path, groups):
    import matplotlib.pyplot as plt

    summary, _ = analyze_intervals(predictions(), groups=groups, n_boot=20)
    before = summary.copy(deep=True)
    figures = plt.get_fignums()
    output = tmp_path / "coverage.png"
    plot_coverage(summary, "smokecast", output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == figures
    pd.testing.assert_frame_equal(summary, before)
    with pytest.raises(FileExistsError):
        plot_coverage(summary, "smokecast", output)


def test_percentile_bounds_outside_point_estimate_are_not_truncated(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    summary, _ = analyze_intervals(predictions(), n_boot=20)
    summary["coverage_ci_lo"], summary["coverage_ci_hi"] = .7, .8
    captured = []
    original = Axes.vlines
    def record(self, x, ymin, ymax, **kwargs):
        captured.append((np.asarray(ymin), np.asarray(ymax)))
        return original(self, x, ymin, ymax, **kwargs)
    monkeypatch.setattr(Axes, "vlines", record)
    plot_coverage(summary, "smokecast", tmp_path / "coverage.png")
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0][0], [.7])
    np.testing.assert_array_equal(captured[0][1], [.8])


@pytest.mark.parametrize("missing", ["bounds", "observations"])
def test_plot_handles_unavailable_bounds_or_score_estimates(tmp_path, missing):
    frame = predictions()
    if missing == "bounds":
        frame = frame.iloc[:1]
    else:
        frame["y_L1"] = np.nan
    summary, _ = analyze_intervals(frame, groups=["point", "smoke"], n_boot=20)
    plot_coverage(summary, "smokecast", tmp_path / "coverage.png")
    assert (tmp_path / "coverage.png").read_bytes().startswith(b"\x89PNG")


def test_unknown_model_is_rejected_before_creating_a_plot(tmp_path):
    summary, _ = analyze_intervals(predictions(), n_boot=20)
    with pytest.raises(ValueError, match="no interval summaries"):
        plot_coverage(summary, "missing", tmp_path / "coverage.png")
    assert not (tmp_path / "coverage.png").exists()


def test_cli_plot_option_records_model_file_mapping(tmp_path):
    frame = predictions()
    frame["other_lo_L1"], frame["other_hi_L1"] = 0., 100.
    source, config, output = (tmp_path / name for name in ("predictions.parquet", "study.yaml", "report"))
    frame.to_parquet(source, index=False)
    config.write_text("split:\n  calibrate: [2023]\n")
    assert main([str(source), "--config", str(config), "--output", str(output), "--plot",
                 "--group", "point", "--group", "smoke", "--n-boot", "20"]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert {item["model"] for item in manifest["plots"]} == {"smokecast", "other"}
    assert len({item["file"] for item in manifest["plots"]}) == len(manifest["plots"])
    for item in manifest["plots"]:
        assert (output / item["file"]).read_bytes().startswith(b"\x89PNG")
