import json

import numpy as np
import pandas as pd
import pytest

from serein.eval.reliability import analyze_reliability, main
from serein.eval.reliability_plots import plot_reliability


def frame():
    return pd.DataFrame({"point_id": "001", "issue_date": pd.date_range("2023-06-01", periods=42),
                         "y_L1": [10., 50.] * 21, "smokecast_prob_L1": [.2, .7] * 21})


@pytest.mark.parametrize("missing", [None, "probability", "observations"])
def test_plot_writes_without_input_mutation_or_global_figures(tmp_path, missing):
    import matplotlib.pyplot as plt

    predictions = frame()
    if missing == "probability":
        predictions["smokecast_prob_L1"] = np.nan
    if missing == "observations":
        predictions["y_L1"] = np.nan
    bins = analyze_reliability(predictions, n_boot=20)["bins"]
    before, figures = bins.copy(deep=True), plt.get_fignums()
    output = tmp_path / "plot.png"
    plot_reliability(bins, "smokecast", output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == figures
    pd.testing.assert_frame_equal(bins, before)
    with pytest.raises(FileExistsError):
        plot_reliability(bins, "smokecast", output)


def test_invalid_plot_population_does_not_create_output(tmp_path):
    bins = analyze_reliability(frame(), n_boot=20)["bins"]
    with pytest.raises(ValueError, match="no pooled"):
        plot_reliability(bins, "missing", tmp_path / "missing.png")
    bins = pd.concat([bins, bins])
    with pytest.raises(ValueError, match="one pooled row"):
        plot_reliability(bins, "smokecast", tmp_path / "duplicate.png")
    assert not list(tmp_path.iterdir())


def test_saved_bounds_are_not_replaced_with_symmetric_bars(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    bins = analyze_reliability(frame(), n_boot=20)["bins"]
    bins["event_rate_ci_lo"], bins["event_rate_ci_hi"] = .4, .6
    captured = []
    original = Axes.vlines
    def record(self, x, ymin, ymax, **kwargs):
        captured.append((np.asarray(ymin), np.asarray(ymax)))
        return original(self, x, ymin, ymax, **kwargs)
    monkeypatch.setattr(Axes, "vlines", record)
    plot_reliability(bins, "smokecast", tmp_path / "bounds.png")
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0][0], [.4, .4])
    np.testing.assert_array_equal(captured[0][1], [.6, .6])


def test_cli_records_multiple_plot_files(tmp_path):
    predictions = frame()
    predictions["other_prob_L1"] = .5
    source, output = tmp_path / "probabilities.parquet", tmp_path / "report"
    predictions.to_parquet(source)
    assert main(["--predictions", str(source), "--out", str(output), "--n-boot", "20", "--plot"]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert [item["model"] for item in manifest["plots"]] == ["other", "smokecast"]
    for item in manifest["plots"]:
        assert (output / item["file"]).read_bytes().startswith(b"\x89PNG")
