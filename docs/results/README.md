# Results snapshot (2026-11-01, dev mode)

A copy of the key outputs so teammates can see results **without** re-downloading
~150 GB of data. Everything here is **development** scoring: trained on 2021-22,
scored on held-out (odd ISO) weeks of 2023. The 2024-25 test seasons have NOT been
scored yet (see "Finishing the proof" in the main README).

| File | What it is |
|---|---|
| `dev_report.md` | Full dev backtest report: RMSE by lead, smoke-day RMSE, warning scores, interval coverage, head-to-head with 95% CIs, comparison with EPA forecasts |
| `dev_holdout_report.md` | Same, with 20% of monitors hidden from the model (forecasting places with no sensor) |
| `metrics.csv` | Every metric for every model, lead and subset |
| `head_to_head.csv` | SmokeCast vs. each baseline on identical rows, with day-bootstrap CIs |
| `vs_epa_forecast.csv` | Category accuracy + warnings vs. EPA's official AQI forecast |
| `peak_report.csv` | How much of each real smoke peak was forecast |
| `impact.txt` | Plain-English impact numbers (warned smoky days, false alarms) |
| `airplan_simulation_summary.csv` | AirPlan synthetic-children study (Brier gain by days logged) |
| `figures/` | Charts and the August 2023 forecast map for the brief, video and website |
| `logs/abl_*.log` | Input ablations (final code). Summary table in `docs/METHODS.md` |
| `logs/peaks_w*.log` | Smoke-day loss-weight experiment (B-13) |
| `logs/sweep_*_reg*.log` | Settings sweep. **Earlier code version** (before smoke weighting and the neighbor-based base), so numbers differ slightly from the final dev run |

To regenerate everything: `./scripts/run_dev_suite.sh` (after downloading data).
