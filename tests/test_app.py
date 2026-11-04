"""Exercise the demo's user flows with a small, local forecast fixture."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("streamlit_folium")
from streamlit.testing.v1 import AppTest

from serein import config

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def demo_app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    maps = tmp_path / "outputs" / "maps"
    maps.mkdir(parents=True)
    issued = date(2023, 8, 18)
    rows = []
    for lead in range(4):
        for kind, lat, lon in [("grid", 38.58, -121.49), ("epa", 38.6, -121.5)]:
            rows.append({
                "lead": lead, "valid_date": issued + timedelta(days=lead),
                "point_id": kind, "kind": kind, "lat": lat, "lon": lon,
                "q05": 5.0, "q25": 10.0, "q50": 20.0, "q75": 30.0,
                "q95": 40.0, "aqi_cat": 1, "obs": 18.0,
            })
    pd.DataFrame(rows).to_parquet(maps / "forecast_20230818.parquet", index=False)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    return app


@pytest.mark.parametrize("days", [0, 1, 2, 120])
def test_family_simulation_supports_full_history_slider_range(demo_app, days):
    demo_app.slider(key="history_days").set_value(days).run()

    assert not demo_app.exception
    assert demo_app.slider(key="history_days").value == days
    assert any("Maya" in element.value for element in demo_app.text)


@pytest.mark.parametrize("language,day_three,day_two", [
    ("en", "in three days", "the day after tomorrow"),
    ("es", "dentro de tres días", "pasado mañana"),
])
def test_day_three_plan_uses_selected_forecast_horizon(demo_app, language, day_three, day_two):
    demo_app.radio(key="forecast_lead").set_value(3)
    demo_app.selectbox(key="language").set_value(language).run()

    assert not demo_app.exception
    message = "\n".join(element.value for element in demo_app.text)
    assert day_three in message
    assert day_two not in message


def test_checkin_can_be_saved_and_cleared_from_the_session(demo_app):
    symptoms = next(radio for radio in demo_app.radio if radio.label == "Any asthma symptoms today?")
    symptoms.set_value(1)
    save = next(button for button in demo_app.button if button.label == "Save today")
    save.click().run()

    assert not demo_app.exception
    assert len(demo_app.session_state["log"]) == 1
    assert demo_app.session_state["log"][0].symptoms == 1
    assert len(demo_app.dataframe) == 1
    assert demo_app.dataframe[0].value["symptoms"].tolist() == [1]

    demo_app.button(key="clear_session_log").click().run()

    assert not demo_app.exception
    assert demo_app.session_state["log"] == []
    assert len(demo_app.dataframe) == 0
