from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from serein.eval.split_plots import plot_split_audit


def assignments():
    return pd.DataFrame({
        "role": ["train", "validation", "validation", "score"],
        "issue_date": pd.to_datetime(["2021-06-01", "2023-06-04", "2023-06-05", "2023-06-08"]),
        "keep_proposed": [True, True, False, True],
    })


def overlap():
    return pd.DataFrame({"role_a": ["validation"], "role_b": ["score"],
                         "target_date": pd.to_datetime(["2023-06-08"]),
                         "n_shared_labels": [12]})


def test_calendar_uses_issue_dates_and_target_dates_separately(monkeypatch, tmp_path):
    captured = []

    def savefig(self, output, **kwargs):
        captured.append(self)
        assert Path(output) == tmp_path / "plot.png"

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", savefig)
    plot_split_audit(assignments(), overlap(), overlap().iloc[:0], tmp_path / "plot.png")
    fig = captured[0]
    assert len(fig.axes) == 4
    assert [ax.get_title() for ax in fig.axes[:2]] == ["2021", "2023"]
    # plt.subplots flattens axes by row: calendars first, then target-count plots.
    calendar = fig.axes[1]
    assert sum(len(collection.get_offsets()) for collection in calendar.collections) == 3
    target = fig.axes[3]
    original, proposed = target.lines
    assert original.get_label() == "validation / score: original"
    offset = (pd.Timestamp("2023-06-08") - pd.Timestamp("2023-01-01")).days
    np.testing.assert_array_equal(np.flatnonzero(original.get_ydata()), [offset])
    assert original.get_ydata()[offset] == 12
    assert target.get_ylim()[1] > original.get_ydata().max()
    assert not proposed.get_ydata().any()
    assert fig.number not in plt.get_fignums()


def test_real_png_and_no_overwrite(tmp_path):
    output = tmp_path / "calendar.png"
    plot_split_audit(assignments(), overlap(), overlap().iloc[:0], output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        plot_split_audit(assignments(), overlap(), overlap(), output)
    assert output.read_bytes() == before


def test_no_overlap_still_shows_each_issue_year(tmp_path):
    empty = overlap().iloc[:0]
    output = tmp_path / "empty.png"
    plot_split_audit(assignments(), empty, empty, output)
    assert output.is_file()


def test_shared_target_in_next_year_gets_its_own_panel(monkeypatch, tmp_path):
    counts = overlap().assign(target_date=pd.to_datetime(["2024-01-01"]))
    captured = []
    monkeypatch.setattr("matplotlib.figure.Figure.savefig", lambda fig, *a, **k: captured.append(fig))
    plot_split_audit(assignments(), counts, counts.iloc[:0], tmp_path / "future.png")
    assert [ax.get_title() for ax in captured[0].axes[:3]] == ["2021", "2023", "2024"]


def test_plot_failure_closes_figure(monkeypatch, tmp_path):
    before = plt.get_fignums()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", fail)
    with pytest.raises(OSError, match="disk full"):
        plot_split_audit(assignments(), overlap(), overlap(), tmp_path / "failed.png")
    assert plt.get_fignums() == before


def test_empty_assignments_rejected(tmp_path):
    with pytest.raises(ValueError, match="requires issue assignments"):
        plot_split_audit(assignments().iloc[:0], overlap(), overlap(), tmp_path / "empty.png")
