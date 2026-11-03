"""Serein demo app.

Run:  uv run serein map --date 2024-09-06 --run west_us_dev     (makes a forecast file)
      uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from serein.aqi import CATEGORIES
from serein.config import REPO_ROOT
from serein.models.infer import AQI_COLORS
from serein.personal.action_plan import make_plan
from serein.personal.model import ChildModel, design_row
from serein.personal.simulate import simulate_children, simulate_log, simulate_pm_season

st.set_page_config(page_title="Serein", layout="wide")
st.title("Serein")
st.caption("Neighborhood wildfire-smoke forecasts with a personal plan for kids with asthma. "
           "Research prototype: not medical advice.")

maps = sorted((REPO_ROOT / "outputs" / "maps").glob("forecast_*.parquet"))
if not maps:
    # Fresh clone: fall back to the sample forecast shipped with the app.
    maps = sorted((Path(__file__).parent / "sample").glob("forecast_*.parquet"))
    if maps:
        st.info("Showing the sample forecast (issued 2023-08-18). Make your own with "
                "`uv run serein map --date YYYY-MM-DD --run <run>`.")
if not maps:
    st.warning("No forecasts yet. Run `uv run serein map --date YYYY-MM-DD --run <run>` first.")
    st.stop()

with st.sidebar:
    f = st.selectbox("Forecast issued", maps, format_func=lambda p: p.stem.replace("forecast_", ""))
    fc = pd.read_parquet(f)
    lead = st.radio("Day", sorted(fc["lead"].unique()),
                    format_func=lambda x: ["today", "tomorrow", "day +2", "day +3"][x])
    lang = st.selectbox("Language", ["en", "es"], format_func=lambda x: {"en": "English", "es": "Español"}[x])

d = fc[fc["lead"] == lead].copy()
tab_map, tab_family, tab_log = st.tabs(["Map", "Family plan", "Daily check-in"])

with tab_map:
    import folium
    from streamlit_folium import st_folium

    m = folium.Map(location=[d["lat"].mean(), d["lon"].mean()], zoom_start=5, tiles="OpenStreetMap")
    step = 0.125
    for _, r in d[d["kind"] == "grid"].iterrows():
        folium.Rectangle([[r.lat - step, r.lon - step], [r.lat + step, r.lon + step]],
                         color=None, fill=True, fill_opacity=0.45,
                         fill_color=AQI_COLORS[int(r.aqi_cat)],
                         tooltip=f"{r.q50:.0f} µg/m³ (90%: {r.q05:.0f}-{r.q95:.0f})").add_to(m)
    for _, r in d[d["kind"] != "grid"].iterrows():
        obs = "" if pd.isna(r.get("obs")) else f"<br>observed: {r.obs:.0f}"
        folium.CircleMarker([r.lat, r.lon], radius=4, weight=1, color="#222", fill=True,
                            fill_color=AQI_COLORS[int(r.aqi_cat)], fill_opacity=1,
                            tooltip=f"sensor {r.point_id}: {r.q50:.0f} µg/m³ "
                                    f"(90%: {r.q05:.0f}-{r.q95:.0f}){obs}").add_to(m)
    st_folium(m, height=650, use_container_width=True)
    st.caption(" · ".join(f"{c}" for c in CATEGORIES))

with tab_family:
    c1, c2 = st.columns([1, 2])
    with c1:
        name = st.text_input("Child's first name", "Maya")
        lat = st.number_input("Home latitude", value=38.58, format="%.3f")
        lon = st.number_input("Home longitude", value=-121.49, format="%.3f")
        seed = st.number_input("Synthetic child (demo log)", 1, 999, 3)
        days = st.slider("Days of symptom log", 0, 120, 60)
        outdoor = st.slider("Time outdoors tomorrow", 0.0, 1.0, 0.3)
        cold = st.checkbox("Has a cold right now")
    near = d.iloc[np.argmin((d["lat"] - lat) ** 2 + ((d["lon"] - lon) * np.cos(np.radians(lat))) ** 2)]
    kid = simulate_children(1, int(seed))[0]
    model = ChildModel()
    if days:
        log = simulate_log(kid, simulate_pm_season(days + 2, int(seed)), int(seed)).iloc[:days]
        X = design_row(log["pm"], log["lag1"], log["lag2"], log["outdoor"], log["cold"], log["missed"])
        model.fit(X, log["y"].to_numpy())
    q = {0.05: near.q05, 0.25: near.q25, 0.5: near.q50, 0.75: near.q75, 0.95: near.q95}
    plan = make_plan(model, name, q, pm_lag1=near.q50, pm_lag2=near.q50, outdoor_frac=outdoor,
                     cold=int(cold), day_label=["today", "tomorrow", "day_after", "day_after"][lead],
                     on=near.valid_date, language=lang)
    with c2:
        color = {"green": "#1e8449", "yellow": "#b7950b", "red": "#c0392b"}[plan.zone]
        st.markdown(f"<h2 style='color:{color}'>{plan.zone.upper()}</h2>", unsafe_allow_html=True)
        st.text(plan.message)
        s = model.sensitivity()
        st.caption(f"Model: odds of symptoms x{s['or_per_10']:.2f} per +10 µg/m³ "
                   f"(90% range {s['lo']:.2f}-{s['hi']:.2f}) after {days} logged days "
                   f"(synthetic child, true type: {kid.kind.replace('_', ' ')}).")

with tab_log:
    from datetime import date as _date

    from serein.personal.log import CheckIn, to_frame

    st.subheader("10-second daily check-in")
    st.caption("Stays on this device. Nothing is uploaded. Demo only: do not enter real health data.")
    if "log" not in st.session_state:
        st.session_state["log"] = []
    with st.form("checkin"):
        day = st.date_input("Day", _date.today())
        sym = st.radio("Any asthma symptoms today?", [0, 1, 2, 3], horizontal=True,
                       format_func=lambda x: ["none", "mild", "moderate", "severe"][x])
        puffs = st.radio("Quick-relief inhaler puffs (not counting before exercise)", [0, 1, 2, 3],
                         horizontal=True, format_func=lambda x: ["0", "1-2", "3-4", "5+"][x])
        night = st.checkbox("Woke up at night because of asthma")
        outdoor = st.radio("Time outdoors", [0, 1, 2], horizontal=True,
                           format_func=lambda x: ["little (<1 h)", "some (1-3 h)", "a lot (3 h+)"][x])
        sick = st.checkbox("Has a cold / is sick")
        missed = st.checkbox("Missed daily controller medicine")
        if st.form_submit_button("Save today"):
            entry = CheckIn(day, sym, puffs, night, outdoor, sick, missed)
            st.session_state["log"].append(entry)
            if entry.needs_doctor_flag:
                st.error("Severe symptoms or 5+ puffs: follow your child's asthma action plan and "
                         "call their doctor. If they struggle to breathe, call 911.")
            else:
                st.success("Saved.")
    log = to_frame(st.session_state["log"])
    if len(log):
        st.dataframe(log, width="stretch")
        st.download_button("Download my log (CSV)", log.to_csv(index=False), "serein_log.csv")
