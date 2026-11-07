import json

import numpy as np
import pandas as pd
import pytest

from serein.eval.discrimination import analyze_discrimination, main
from serein.eval.discrimination_plots import plot_discrimination


def frame():
    return pd.DataFrame({"point_id": "001", "issue_date": pd.date_range("2023-06-01", periods=42),
                         "y_L1": [10., 50.] * 21, "smokecast_prob_L1": [.2, .7] * 21})


@pytest.mark.parametrize("kind", ["mixed", "no_events", "all_events", "missing_probabilities", "missing_targets"])
def test_plot_does_not_mutate_tables_or_leak_figures(tmp_path, kind):
    import matplotlib.pyplot as plt

    predictions = frame()
    if kind == "no_events":
        predictions["y_L1"] = 0.
    elif kind == "all_events":
        predictions["y_L1"] = 50.
    elif kind == "missing_probabilities":
        predictions["smokecast_prob_L1"] = np.nan
    elif kind == "missing_targets":
        predictions["y_L1"] = np.nan
    result = analyze_discrimination(predictions, n_boot=20)
    before = {name: table.copy(deep=True) for name, table in result.items()}
    figures = plt.get_fignums()
    output = tmp_path / "plot.png"
    plot_discrimination(result, "smokecast", output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == figures
    for name in result:
        pd.testing.assert_frame_equal(result[name], before[name])
    with pytest.raises(FileExistsError):
        plot_discrimination(result, "smokecast", output)


def test_plot_uses_saved_scores_and_warn_none_display_endpoint(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    result = analyze_discrimination(frame(), n_boot=20)
    captured = []
    original = Axes.step
    def record(self, x, y, **kwargs):
        captured.append((np.asarray(x), np.asarray(y), kwargs))
        return original(self, x, y, **kwargs)
    monkeypatch.setattr(Axes, "step", record)
    plot_discrimination(result, "smokecast", tmp_path / "plot.png")
    assert len(captured) == 1
    assert captured[0][0].tolist() == [0., 1., 1.]
    assert captured[0][1].tolist() == [1., 1., .5]
    assert captured[0][2]["where"] == "pre"
    assert "AP 1.000" in captured[0][2]["label"]


def test_invalid_plot_populations_create_no_output(tmp_path):
    result = analyze_discrimination(frame(), n_boot=20)
    output = tmp_path / "plot.png"
    with pytest.raises(ValueError, match="one pooled summary"):
        plot_discrimination(result, "missing", output)
    duplicate = {**result, "summary": pd.concat([result["summary"]] * 2)}
    with pytest.raises(ValueError, match="one pooled summary"):
        plot_discrimination(duplicate, "smokecast", output)
    duplicate = {**result, "curves": pd.concat([result["curves"]] * 2)}
    with pytest.raises(ValueError, match="unique pooled curve"):
        plot_discrimination(duplicate, "smokecast", output)
    missing = {**result, "curves": result["curves"].iloc[:0]}
    with pytest.raises(ValueError, match="no pooled curve"):
        plot_discrimination(missing, "smokecast", output)
    assert not output.exists()


def test_cli_multiple_models_and_plots(tmp_path):
    predictions = frame()
    predictions["other_prob_L1"] = .5
    source, output = tmp_path / "input.parquet", tmp_path / "report"
    predictions.to_parquet(source)
    assert main(["--predictions", str(source), "--out", str(output), "--n-boot", "20", "--plot"]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert [x["model"] for x in manifest["plots"]] == ["other", "smokecast"]
    for item in manifest["plots"]:
        assert (output / item["file"]).read_bytes().startswith(b"\x89PNG")
