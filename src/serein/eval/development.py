"""Run the development study from Python, with isolated outputs and step logs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from serein.config import REPO_ROOT, load_study

ABLATIONS = (
    ("full", ()),
    ("no_wind", ("--no-wind",)),
    ("no_hrrr", ("--drop", "hrrr_,wind_now_")),
    ("no_aqm", ("--drop", "aqm_")),
    ("no_fires", ("--drop", "frp_,fire_dist")),
    ("no_hms", ("--drop", "hms_prev")),
    ("no_sensors", ("--drop", "obs_")),
)


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    root: Path
    log_dir: Path
    run_dirs: tuple[Path, ...]
    steps: tuple[Step, ...]
    timeout_seconds: float | None = None

    def as_dict(self) -> dict:
        return {"root": str(self.root), "log_dir": str(self.log_dir),
                "timeout_seconds": self.timeout_seconds,
                "run_dirs": [str(path) for path in self.run_dirs],
                "steps": [{"name": step.name, "command": list(step.command)}
                          for step in self.steps]}


def _identifier(value: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError(f"{name} must be 1 to 80 letters, digits, underscores or hyphens")
    return value


def make_plan(
    config: Path, tag: str, *, suite: str = "all", epochs: int = 300,
    seeds: int = 5, ablation_seeds: int = 3, map_date: date | None = None,
    timeout_minutes: float | None = None,
    root: Path = REPO_ROOT,
) -> Plan:
    """Build only development commands. Creating a plan performs no writes."""
    root = root.resolve()
    config = (root / config).resolve()
    study = load_study(config)
    _identifier(study.name, "study name")
    _identifier(tag, "tag")
    if suite not in {"all", "dev", "ablations"}:
        raise ValueError("suite must be all, dev or ablations")
    for name, value in (("epochs", epochs), ("seeds", seeds), ("ablation_seeds", ablation_seeds)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if timeout_minutes is not None and (
        isinstance(timeout_minutes, bool) or not math.isfinite(timeout_minutes) or timeout_minutes <= 0
    ):
        raise ValueError("timeout_minutes must be finite and positive")
    if map_date is not None:
        if suite == "ablations":
            raise ValueError("a map requires the dev suite")
        if (map_date.year not in study.raw["split"]["calibrate"]
                or map_date.month not in study.raw["season_months"]):
            raise ValueError("map date must fall in a configured calibration season")
    python = sys.executable
    cli = (python, "-m", "serein.cli")
    common = ("--config", str(config), "--epochs", str(epochs))
    runs, steps = [], []
    log_dir = root / "outputs" / "logs" / f"{study.name}_{tag}"

    def backtest(name, run_tag, options):
        run = root / "outputs" / "backtest" / f"{study.name}_{run_tag}"
        runs.append(run)
        steps.append(Step(name, cli + ("backtest",) + common + ("--tag", run_tag) + options))
        return run

    if suite in {"all", "dev"}:
        run = backtest("dev", tag, ("--seeds", str(seeds)))
        steps.extend([
            Step("figures", (python, "-m", "serein.eval.figures", str(run))),
            Step("impact", (python, "-m", "serein.eval.impact", str(run))),
            Step("comparison", (python, "-m", "serein.eval.comparison", str(run),
                                "--output", str(log_dir / "comparison.csv"),
                                "--daily-output", str(log_dir / "comparison_daily.csv"))),
        ])
        if map_date is not None:
            steps.append(Step("map", (python, "-m", "serein.eval.development", "map",
                                     "--config", str(config), "--date", map_date.isoformat(),
                                     "--run", str(run), "--output", str(run / "maps"))))
        backtest("holdout", f"{tag}_holdout",
                 ("--seeds", str(seeds), "--holdout-sites", "0.2", "--no-gbm"))
    if suite in {"all", "ablations"}:
        for name, options in ABLATIONS:
            backtest(f"abl_{name}", f"{tag}_abl_{name}",
                     ("--no-gbm", "--no-ablation", "--seeds", str(ablation_seeds)) + options)
    return Plan(root, log_dir, tuple(runs), tuple(steps),
                None if timeout_minutes is None else timeout_minutes * 60)


def run_plan(plan: Plan) -> int:
    """Stop at the first failed command, retaining its output and a manifest."""
    destinations = (plan.log_dir,) + plan.run_dirs
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"refusing existing output directory: {path}")
    # Reserve all run folders before launching any training process.
    for path in destinations:
        path.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    manifest = {**plan.as_dict(), "started_utc": _now(), "results": [], "status": "running"}
    exit_code = 0
    deadline = None if plan.timeout_seconds is None else time.monotonic() + plan.timeout_seconds
    timed_out = False
    try:
        for step in plan.steps:
            log = plan.log_dir / f"{step.name}.log"
            record = {"name": step.name, "log": str(log), "started_utc": _now()}
            manifest["results"].append(record)
            print(f"{step.name}: started; log {log}", flush=True)
            try:
                with log.open("x", encoding="utf-8", newline="") as stream:
                    stream.write(json.dumps(list(step.command)) + "\n")
                    stream.flush()
                    remaining = None if deadline is None else max(0, deadline - time.monotonic())
                    if remaining == 0:
                        raise subprocess.TimeoutExpired(step.command, 0)
                    completed = subprocess.run(step.command, cwd=plan.root, env=env,
                                               stdout=stream, stderr=subprocess.STDOUT,
                                               check=False, timeout=remaining)
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                record["error"] = "suite time limit exceeded"
                exit_code = 124
            except OSError as exc:
                record["error"] = str(exc)
                exit_code = 127
            record.update(exit_code=exit_code, finished_utc=_now())
            if exit_code:
                print(f"{step.name}: FAILED (exit {exit_code}); see {log}", file=sys.stderr,
                      flush=True)
                break
            print(f"{step.name}: done", flush=True)
        manifest["status"] = "timed_out" if timed_out else ("failed" if exit_code else "completed")
    except KeyboardInterrupt:
        exit_code = 130
        manifest["status"] = "interrupted"
        if manifest["results"]:
            manifest["results"][-1].update(exit_code=exit_code, finished_utc=_now())
        print("Development suite interrupted", file=sys.stderr, flush=True)
    except Exception as exc:
        exit_code = 1
        manifest.update(status="failed", error=str(exc))
        if manifest["results"]:
            manifest["results"][-1].update(exit_code=exit_code, finished_utc=_now())
        raise
    finally:
        manifest.update(finished_utc=_now(), exit_code=exit_code)
        with (plan.log_dir / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    if not exit_code:
        print(f"ALLDONE: {plan.log_dir}", flush=True)
    return exit_code


def _now() -> str:
    return datetime.now(UTC).isoformat()


def render_map(config: Path, day: date, run: Path, output: Path) -> None:
    """Save a development map beside its model, preserving the shared map folder."""
    study = load_study(config)
    if (day.year not in study.raw["split"]["calibrate"]
            or day.month not in study.raw["season_months"]):
        raise ValueError("map date must fall in a configured calibration season")
    if not run.is_dir():
        raise ValueError(f"model run directory does not exist: {run}")
    output.mkdir(parents=True, exist_ok=False)
    from serein.models.infer import forecast_day, plot_map

    forecast = forecast_day(study, day, run)
    forecast.to_parquet(output / f"forecast_{day:%Y%m%d}.parquet", index=False)
    plot_map(forecast, output / f"map_{day:%Y%m%d}.png")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "run"):
        sp = sub.add_parser(action, help="show commands" if action == "plan" else "execute commands")
        sp.add_argument("--config", type=Path, default=Path("configs/west_us.yaml"))
        sp.add_argument("--tag", default=datetime.now().strftime("native_%Y%m%d_%H%M%S_%f"))
        sp.add_argument("--suite", choices=("all", "dev", "ablations"), default="all")
        sp.add_argument("--epochs", type=int, default=300)
        sp.add_argument("--seeds", type=int, default=5)
        sp.add_argument("--ablation-seeds", type=int, default=3)
        sp.add_argument("--map-date", type=date.fromisoformat)
        sp.add_argument("--timeout-minutes", type=float, help="time limit for the entire suite")
    sp = sub.add_parser("map", help="render a saved development model into a new directory")
    sp.add_argument("--config", type=Path, required=True)
    sp.add_argument("--date", type=date.fromisoformat, required=True)
    sp.add_argument("--run", type=Path, required=True)
    sp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "map":
            render_map(args.config, args.date, args.run, args.output)
            return 0
        plan = make_plan(args.config, args.tag, suite=args.suite, epochs=args.epochs,
                         seeds=args.seeds, ablation_seeds=args.ablation_seeds,
                         map_date=args.map_date, timeout_minutes=args.timeout_minutes)
        if args.action == "plan":
            print(json.dumps(plan.as_dict(), indent=2))
            return 0
        return run_plan(plan)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
