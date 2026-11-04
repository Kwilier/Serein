"""Synthetic children: test AirPlan with zero real health data (TODO C-2, C-6).

We invent children whose TRUE smoke sensitivity we know, simulate a smoke season
and their daily symptoms, then check whether AirPlan can figure each child out
from their log, and how many days of logging that takes.

The simulated cohort is deliberately more varied than the prior assumes
(some children very sensitive, some barely), because that is exactly the case
where personal plans should beat one-size-fits-all advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from serein.config import REPO_ROOT
from serein.personal.model import FEATURES, ChildModel, PopulationPrior, design_row, sigmoid


# ------------------------------------------------------------------ smoke season
def simulate_pm_season(n_days: int = 150, seed: int = 0) -> np.ndarray:
    """Daily PM2.5: clean background (AR(1) around ~7 µg/m³) plus smoke episodes."""
    if (isinstance(n_days, (bool, np.bool_))
            or not isinstance(n_days, Integral) or n_days < 1):
        raise ValueError("n_days must be a positive integer")
    rng = np.random.default_rng(seed)
    z = np.zeros(n_days)
    for t in range(1, n_days):
        z[t] = 0.7 * z[t - 1] + rng.normal(0, 0.25)
    pm = 7.0 * np.exp(z)
    for _ in range(rng.poisson(3) + 1):
        start = rng.integers(0, max(1, n_days - 3))
        length = rng.integers(2, 11)
        peak = rng.choice([25, 45, 80, 150, 250], p=[0.3, 0.3, 0.2, 0.15, 0.05])
        shape = np.sin(np.linspace(0, np.pi, length)) ** 0.7
        end = min(start + length, n_days)
        pm[start:end] += peak * shape[: end - start] * rng.uniform(0.6, 1.2, end - start)
    return pm


# --------------------------------------------------------------------- children
@dataclass
class SimChild:
    beta: np.ndarray
    outdoor_mean: float
    adherence: float
    kind: str

    def __post_init__(self):
        # Coefficients not given (e.g. ozone, heat, pollen) mean "no effect".
        self.beta = np.r_[self.beta, np.zeros(len(FEATURES) - len(self.beta))]


def simulate_children(n: int, seed: int = 0) -> list[SimChild]:
    rng = np.random.default_rng(seed)
    kids = []
    for _ in range(n):
        kind = rng.choice(["very_sensitive", "typical", "not_sensitive"], p=[0.25, 0.5, 0.25])
        # Typical children match the literature average (~1.04/10 µg/m³); the
        # "very sensitive" tail is an assumption the personal model must discover.
        or10 = {"very_sensitive": rng.uniform(1.3, 1.8), "typical": rng.uniform(1.02, 1.10),
                "not_sensitive": rng.uniform(0.98, 1.01)}[kind]
        base = rng.uniform(0.05, 0.20)
        beta = np.array([np.log(base / (1 - base)), np.log(or10), np.log(or10) * 0.6,
                         np.log(or10) * 0.3, np.log(rng.uniform(2, 4.5)), np.log(rng.uniform(1.2, 2.0))])
        kids.append(SimChild(beta, rng.uniform(0.1, 0.6), rng.uniform(0.6, 0.98), kind))
    return kids


def simulate_log(child: SimChild, pm: np.ndarray, seed: int = 0,
                 forecast_error_sd: float = 0.35) -> pd.DataFrame:
    """One child's season: covariates, symptoms (0/1), and the smoke FORECAST they'd see."""
    rng = np.random.default_rng(seed)
    n = len(pm)
    cold = np.zeros(n)
    t = 0
    while t < n:
        if rng.random() < 0.025:
            cold[t:t + rng.integers(5, 10)] = 1
            t += 10
        t += 1
    outdoor = np.clip(rng.normal(child.outdoor_mean, 0.15, n), 0, 1)
    missed = (rng.random(n) > child.adherence).astype(float)
    lag1 = pm[np.maximum(np.arange(n) - 1, 0)]
    lag2 = pm[np.maximum(np.arange(n) - 2, 0)]
    X = design_row(pm, lag1, lag2, outdoor, cold, missed)
    y = (rng.random(n) < sigmoid(X @ child.beta)).astype(float)
    fc = pm * np.exp(rng.normal(0, forecast_error_sd, n))  # forecast for that day
    return pd.DataFrame({"pm": pm, "pm_forecast": fc, "lag1": lag1, "lag2": lag2,
                         "outdoor": outdoor, "cold": cold, "missed": missed, "y": y})


def _pred(model: ChildModel, X: np.ndarray) -> np.ndarray:
    """Posterior predictive probability (MacKay's probit approximation; fast)."""
    m, C = model._params()
    mu = X @ m
    var = np.einsum("ij,jk,ik->i", X, C, X)
    return sigmoid(mu / np.sqrt(1 + np.pi * var / 8))


def run_study(n_children: int = 300, n_days: int = 150, eval_days: int = 50,
              ks=(0, 7, 14, 21, 30, 45, 60, 100), seed: int = 0) -> pd.DataFrame:
    """How well does AirPlan predict a child's symptom days after k days of logging?

    Scores on the LAST ``eval_days`` days (never used for fitting), using the
    smoke FORECAST (with error), as it would be live.
    k=0 is the one-size-fits-all population model. 'oracle' knows the true β.
    Durations above the fitting window use all available fitting days; each
    effective duration is scored once, in first-requested order.
    """
    for name, value, minimum in (("n_children", n_children, 1), ("n_days", n_days, 4),
                                 ("eval_days", eval_days, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if eval_days >= n_days:
        raise ValueError("eval_days must be smaller than n_days")
    ks = tuple(ks)
    if not ks or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 0
                     for k in ks):
        raise ValueError("ks must contain nonnegative integer logging durations")
    ks = tuple(dict.fromkeys(min(k, n_days - eval_days) for k in ks))
    kids = simulate_children(n_children, seed)
    rows = []
    for i, kid in enumerate(kids):
        pm = simulate_pm_season(n_days, seed * 10_000 + i)
        log = simulate_log(kid, pm, seed=seed * 10_000 + i)
        fit_part, ev = log.iloc[: n_days - eval_days], log.iloc[n_days - eval_days:]
        Xe = design_row(ev["pm_forecast"], ev["lag1"], ev["lag2"], ev["outdoor"], ev["cold"], ev["missed"])
        ye = ev["y"].to_numpy()
        Xo = design_row(ev["pm"], ev["lag1"], ev["lag2"], ev["outdoor"], ev["cold"], ev["missed"])
        oracle_p = sigmoid(Xo @ kid.beta)
        for k in ks:
            m = ChildModel(PopulationPrior())
            if k > 0:
                f = fit_part.iloc[:k]
                Xf = design_row(f["pm"], f["lag1"], f["lag2"], f["outdoor"], f["cold"], f["missed"])
                m.fit(Xf, f["y"].to_numpy())
            p = np.clip(_pred(m, Xe), 1e-4, 1 - 1e-4)
            s = m.sensitivity()
            rows.append({"child": i, "kind": kid.kind, "k": k,
                         "logloss": float(-np.mean(ye * np.log(p) + (1 - ye) * np.log(1 - p))),
                         "brier": float(np.mean((p - ye) ** 2)),
                         "true_or": float(np.exp(kid.beta[1])), "est_or": s["or_per_10"],
                         "or_lo": s["lo"], "or_hi": s["hi"]})
        po = np.clip(oracle_p, 1e-4, 1 - 1e-4)
        rows.append({"child": i, "kind": kid.kind, "k": "oracle",
                     "logloss": float(-np.mean(ye * np.log(po) + (1 - ye) * np.log(1 - po))),
                     "brier": float(np.mean((po - ye) ** 2)),
                     "true_or": float(np.exp(kid.beta[1])), "est_or": float(np.exp(kid.beta[1])),
                     "or_lo": np.nan, "or_hi": np.nan})
    return pd.DataFrame(rows)


def summarize_study(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize learning curves, requiring a k=0 baseline for every scored child."""
    d = df[df["k"] != "oracle"].copy()
    durations = []
    for value in d["k"]:
        try:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, str)):
                raise TypeError
            duration = int(value)
            if duration < 0 or (not isinstance(value, str) and duration != value):
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise ValueError("study k must contain nonnegative integer logging durations") from None
        durations.append(duration)
    d["k"] = durations
    if d.duplicated(["child", "k"]).any():
        raise ValueError("duplicate child-duration rows in study results")
    pop = d[d["k"] == 0].set_index("child")["brier"]
    if not d["child"].isin(pop.index).all():
        raise ValueError("population baseline (k=0) is required for every child")
    d["brier_gain_vs_population"] = 1 - d["brier"] / d["child"].map(pop)
    d["or_covered"] = (d["true_or"] >= d["or_lo"]) & (d["true_or"] <= d["or_hi"])
    d["or_abs_err"] = np.abs(np.log(d["est_or"]) - np.log(d["true_or"]))
    g = d.groupby(["k", "kind"]).agg(brier=("brier", "mean"),
                                     gain=("brier_gain_vs_population", "mean"),
                                     or_log_err=("or_abs_err", "mean"),
                                     or_ci_coverage=("or_covered", "mean")).reset_index()
    return g


def demo(seed: int = 1, days: int = 60, language: str = "en") -> None:
    """Simulate one child, learn from `days` of logs, print tomorrow's plan."""
    from serein.personal.action_plan import make_plan

    if isinstance(days, (bool, np.bool_)) or not isinstance(days, Integral) or days < 0:
        raise ValueError("days must be a nonnegative integer")
    kid = simulate_children(1, seed)[0]
    pm = simulate_pm_season(days + 5, seed)
    log = simulate_log(kid, pm, seed)
    f = log.iloc[:days]
    X = design_row(f["pm"], f["lag1"], f["lag2"], f["outdoor"], f["cold"], f["missed"])
    m = ChildModel().fit(X, f["y"].to_numpy())
    s = m.sensitivity()
    print(f"Simulated child: {kid.kind}, true OR per +10 µg/m³ = {np.exp(kid.beta[1]):.2f}")
    print(f"After {days} logged days AirPlan estimates OR = {s['or_per_10']:.2f} "
          f"(90% range {s['lo']:.2f}-{s['hi']:.2f}); personal threshold ≈ "
          f"{m.personal_threshold():.0f} µg/m³\n")
    for name, q in [("clean day", {0.05: 4, 0.5: 7, 0.95: 12}),
                    ("light smoke", {0.05: 12, 0.5: 22, 0.95: 40}),
                    ("heavy smoke", {0.05: 40, 0.5: 75, 0.95: 140})]:
        # Scenario: the same smoke level for the 2 previous days and tomorrow.
        plan = make_plan(m, "Maya", q, pm_lag1=q[0.5], pm_lag2=q[0.5], language=language)
        print(f"===== {name}: zone {plan.zone.upper()} ({plan.reason}) =====")
        print(plan.message)
        print()


def main_study(out_dir=None, seed: int = 0) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or REPO_ROOT / "outputs" / "airplan"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run_study(seed=seed)
    df.to_csv(out_dir / "simulation_raw.csv", index=False)
    s = summarize_study(df)
    s.to_csv(out_dir / "simulation_summary.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for kind, color in [("very_sensitive", "#2a78d6"), ("typical", "#eb6834"), ("not_sensitive", "#1baf7a")]:
        z = s[s["kind"] == kind]
        ax[0].plot(z["k"], 100 * z["gain"], "o-", color=color, label=kind.replace("_", " "))
        ax[1].plot(z["k"], z["or_log_err"], "o-", color=color, label=kind.replace("_", " "))
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(xlabel="days of symptom log", ylabel="% better than one-size-fits-all (Brier)",
              title="Personal model vs. population model")
    ax[1].set(xlabel="days of symptom log", ylabel="error in estimated sensitivity (|log OR|)",
              title="Learning each child's smoke sensitivity")
    for a in ax:
        a.legend(frameon=False)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "airplan_learning_curve.png", dpi=150)
    return s
