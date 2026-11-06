import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from serein.eval import development
from serein.eval.development import Plan, Step, make_plan, render_map, run_plan


@pytest.fixture
def study_config(tmp_path):
    path = tmp_path / "study with spaces.yaml"
    path.write_text(yaml.safe_dump({
        "name": "synthetic", "season_months": [7, 8],
        "split": {"train": [2021, 2022], "calibrate": [2023], "test": [2024, 2025]},
    }), encoding="utf-8")
    return path


def test_full_plan_preserves_development_study_and_ablation_settings(tmp_path, study_config):
    plan = make_plan(study_config, "local", root=tmp_path, map_date=date(2023, 8, 18))
    assert [step.name for step in plan.steps] == [
        "dev", "figures", "impact", "comparison", "map", "holdout", "abl_full",
        "abl_no_wind", "abl_no_hrrr", "abl_no_aqm", "abl_no_fires", "abl_no_hms",
        "abl_no_sensors",
    ]
    assert not (tmp_path / "outputs").exists()
    assert len(plan.run_dirs) == 9
    assert len(set(plan.run_dirs)) == 9
    for step in plan.steps:
        assert step.command[0] == sys.executable
        assert "--final" not in step.command
        assert "--force" not in step.command
        assert "transfer" not in step.command
        assert "download" not in step.command
    backtests = [step for step in plan.steps if "backtest" in step.command]
    assert len(backtests) == 9
    for step in backtests:
        command = step.command
        assert command[:4] == (sys.executable, "-m", "serein.cli", "backtest")
        assert command[command.index("--config") + 1] == str(study_config)
        assert command[command.index("--epochs") + 1] == "300"
        assert command[command.index("--seeds") + 1] == ("3" if step.name.startswith("abl_")
                                                        else "5")
    holdout = next(step.command for step in plan.steps if step.name == "holdout")
    assert holdout[holdout.index("--holdout-sites") + 1] == "0.2"
    assert "--no-gbm" in holdout
    expected_drops = {"abl_no_hrrr": "hrrr_,wind_now_", "abl_no_aqm": "aqm_",
                      "abl_no_fires": "frp_,fire_dist", "abl_no_hms": "hms_prev",
                      "abl_no_sensors": "obs_"}
    for step in plan.steps:
        if step.name.startswith("abl_"):
            assert "--no-ablation" in step.command
            assert "--no-gbm" in step.command
        if step.name in expected_drops:
            assert step.command[step.command.index("--drop") + 1] == expected_drops[step.name]
    no_wind = next(step for step in plan.steps if step.name == "abl_no_wind")
    assert "--no-wind" in no_wind.command


@pytest.mark.parametrize("suite,steps,runs", [("dev", 5, 2), ("ablations", 7, 7)])
def test_partial_suite_has_only_requested_work(tmp_path, study_config, suite, steps, runs):
    plan = make_plan(study_config, "local", suite=suite, root=tmp_path)
    assert len(plan.steps) == steps
    assert len(plan.run_dirs) == runs
    assert "map" not in [step.name for step in plan.steps]


def test_plan_explicit_settings_reach_each_training_command(tmp_path, study_config):
    plan = make_plan(study_config, "short", epochs=2, seeds=1, ablation_seeds=2, root=tmp_path)
    for step in plan.steps:
        if "backtest" in step.command:
            assert step.command[step.command.index("--epochs") + 1] == "2"
            assert step.command[step.command.index("--seeds") + 1] == (
                "2" if step.name.startswith("abl_") else "1")


@pytest.mark.parametrize("tag", ["", "../other", "has space", "x/y", "x\\y", "-x", "a" * 81])
def test_plan_rejects_unsafe_or_invalid_run_tags(tmp_path, study_config, tag):
    with pytest.raises(ValueError, match="tag"):
        make_plan(study_config, tag, root=tmp_path)
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("options", [
    {"epochs": 0}, {"epochs": True}, {"seeds": -1}, {"seeds": 1.5},
    {"ablation_seeds": 0}, {"suite": "unknown"},
    {"map_date": date(2024, 8, 18)}, {"map_date": date(2025, 8, 18)},
    {"map_date": date(2023, 1, 1)},
    {"suite": "ablations", "map_date": date(2023, 8, 18)},
    {"timeout_minutes": 0}, {"timeout_minutes": -1}, {"timeout_minutes": True},
    {"timeout_minutes": float("inf")}, {"timeout_minutes": float("nan")},
])
def test_invalid_plan_cannot_create_output_directories(tmp_path, study_config, options):
    with pytest.raises(ValueError):
        make_plan(study_config, "local", root=tmp_path, **options)
    assert not (tmp_path / "outputs").exists()


def subprocess_plan(tmp_path, failed_step=None):
    steps = []
    for index in range(4):
        code = ("import os,sys; "
                f"print('step {index} output'); print('step {index} error', file=sys.stderr); "
                "print(os.environ['PYTHONIOENCODING']); "
                "print(os.getcwd()); "
                f"sys.exit({7 if index == failed_step else 0})")
        steps.append(Step(f"step_{index}", (sys.executable, "-c", code)))
    return Plan(tmp_path, tmp_path / "logs", (tmp_path / "run",), tuple(steps))


@pytest.mark.parametrize("failed_step", range(4))
def test_real_child_failure_stops_later_commands_and_records_evidence(tmp_path, capsys, failed_step):
    plan = subprocess_plan(tmp_path, failed_step)
    assert run_plan(plan) == 7
    output = capsys.readouterr()
    assert "ALLDONE" not in output.out
    assert f"step_{failed_step}: FAILED (exit 7)" in output.err
    manifest = json.loads((plan.log_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == 7
    assert len(manifest["results"]) == failed_step + 1
    assert manifest["results"][-1]["exit_code"] == 7
    log = (plan.log_dir / f"step_{failed_step}.log").read_text(encoding="utf-8")
    assert f"step {failed_step} output" in log
    assert f"step {failed_step} error" in log
    assert "utf-8" in log
    assert str(tmp_path) in log
    for index in range(failed_step + 1, 4):
        assert not (plan.log_dir / f"step_{index}.log").exists()


def test_all_real_child_commands_finish_before_success(tmp_path, capsys):
    plan = subprocess_plan(tmp_path)
    assert run_plan(plan) == 0
    output = capsys.readouterr()
    assert output.out.rstrip().endswith(f"ALLDONE: {plan.log_dir}")
    assert output.err == ""
    manifest = json.loads((plan.log_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert len(manifest["results"]) == len(plan.steps)
    assert all(row["exit_code"] == 0 for row in manifest["results"])
    assert manifest["steps"] == plan.as_dict()["steps"]
    assert manifest["started_utc"].endswith("+00:00")
    assert manifest["finished_utc"].endswith("+00:00")


@pytest.mark.parametrize("existing", ["logs", "run"])
def test_existing_output_prevents_any_work_or_overwrite(tmp_path, existing):
    plan = subprocess_plan(tmp_path)
    target = tmp_path / existing
    target.mkdir()
    original = target / "keep.txt"
    original.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing existing"):
        run_plan(plan)
    assert original.read_text(encoding="utf-8") == "original"
    assert not (tmp_path / ("run" if existing == "logs" else "logs")).exists()


def test_unlaunchable_command_reports_failure_and_keeps_manifest(tmp_path, capsys):
    plan = Plan(tmp_path, tmp_path / "logs", (),
                (Step("missing", (str(tmp_path / "not-an-executable"),)),))
    assert run_plan(plan) == 127
    assert "ALLDONE" not in capsys.readouterr().out
    manifest = json.loads((plan.log_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["results"][0]["error"]


def test_keyboard_interrupt_records_partial_progress(tmp_path, monkeypatch, capsys):
    plan = subprocess_plan(tmp_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(development.subprocess, "run", interrupt)
    assert run_plan(plan) == 130
    assert "ALLDONE" not in capsys.readouterr().out
    manifest = json.loads((plan.log_dir / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert len(manifest["results"]) == 1
    assert manifest["results"][0]["exit_code"] == 130


def test_unexpected_failure_does_not_leave_a_successful_manifest(tmp_path, monkeypatch):
    plan = subprocess_plan(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("unexpected subprocess failure")

    monkeypatch.setattr(development.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="unexpected subprocess failure"):
        run_plan(plan)
    manifest = json.loads((plan.log_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == 1
    assert manifest["error"] == "unexpected subprocess failure"
    assert manifest["results"][0]["exit_code"] == 1


def test_suite_timeout_stops_its_child_and_prevents_later_steps(tmp_path, capsys):
    marker = tmp_path / "finished.txt"
    command = (sys.executable, "-c",
               ("import pathlib,time; time.sleep(2); "
                f"pathlib.Path({str(marker)!r}).write_text('finished')"))
    plan = Plan(tmp_path, tmp_path / "logs", (),
                (Step("slow", command), Step("later", command)), timeout_seconds=0.1)
    assert run_plan(plan) == 124
    assert not marker.exists()
    assert not (plan.log_dir / "later.log").exists()
    assert "ALLDONE" not in capsys.readouterr().out
    manifest = json.loads((plan.log_dir / "manifest.json").read_text())
    assert manifest["status"] == "timed_out"
    assert manifest["results"][0]["exit_code"] == 124
    assert manifest["results"][0]["error"] == "suite time limit exceeded"


def test_deadline_is_shared_across_steps_not_reset_per_command(tmp_path, monkeypatch):
    plan = replace(subprocess_plan(tmp_path), timeout_seconds=1)
    monotonic_values = iter([0., 0.25, 0.75, 1.25])
    monkeypatch.setattr(development.time, "monotonic", lambda: next(monotonic_values))
    timeouts = []

    def finish(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(development.subprocess, "run", finish)
    assert run_plan(plan) == 124
    assert timeouts == [0.75, 0.25]
    manifest = json.loads((plan.log_dir / "manifest.json").read_text())
    assert manifest["status"] == "timed_out"
    assert [row["name"] for row in manifest["results"]] == ["step_0", "step_1", "step_2"]


def test_plan_records_converted_suite_time_limit(tmp_path, study_config):
    plan = make_plan(study_config, "limited", root=tmp_path, timeout_minutes=2.5)
    assert plan.timeout_seconds == pytest.approx(2.5 * 60)
    assert plan.as_dict()["timeout_seconds"] == plan.timeout_seconds


def test_runner_preserves_parent_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    plan = Plan(tmp_path, tmp_path / "logs", (),
                (Step("unicode", (sys.executable, "-c", "print('\u00b5g/m\u00b3')")),))
    assert run_plan(plan) == 0
    assert "\u00b5g/m\u00b3" in (plan.log_dir / "unicode.log").read_text(encoding="utf-8")
    assert os.environ["PYTHONIOENCODING"] == "ascii"


def test_plan_command_works_from_another_working_directory(tmp_path, study_config):
    result = subprocess.run(
        [sys.executable, "-m", "serein.eval.development", "plan", "--config", str(study_config),
         "--tag", "cli_probe", "--suite", "dev"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    plan = json.loads(result.stdout)
    assert len(plan["steps"]) == 5
    assert plan["steps"][0]["name"] == "dev"
    assert not (Path(plan["log_dir"])).exists()


def test_map_uses_new_run_specific_directory(tmp_path, study_config, monkeypatch):
    from serein.models import infer

    run = tmp_path / "model"
    run.mkdir()
    target = run / "maps"
    forecast = pd.DataFrame({"point_id": ["synthetic"], "q50": [12.]})
    calls = []

    def forecast_day(study, day, model_run):
        calls.append((study.name, day, model_run))
        return forecast

    def plot_map(frame, path):
        pd.testing.assert_frame_equal(frame, forecast)
        path.write_bytes(b"synthetic plot")

    monkeypatch.setattr(infer, "forecast_day", forecast_day)
    monkeypatch.setattr(infer, "plot_map", plot_map)
    day = date(2023, 8, 18)
    render_map(study_config, day, run, target)
    assert calls == [("synthetic", day, run)]
    pd.testing.assert_frame_equal(pd.read_parquet(target / "forecast_20230818.parquet"), forecast)
    assert (target / "map_20230818.png").read_bytes() == b"synthetic plot"


def test_map_rejects_existing_output_before_inference(tmp_path, study_config, monkeypatch):
    from serein.models import infer

    def forbidden(*args, **kwargs):
        pytest.fail("inference must not run before the output check")

    monkeypatch.setattr(infer, "forecast_day", forbidden)
    run, target = tmp_path / "model", tmp_path / "maps"
    run.mkdir()
    target.mkdir()
    with pytest.raises(FileExistsError):
        render_map(study_config, date(2023, 8, 18), run, target)


@pytest.mark.parametrize("day", [date(2024, 8, 18), date(2025, 8, 18), date(2023, 1, 1)])
def test_map_rejects_dates_outside_development_before_creating_outputs(tmp_path, study_config, day):
    run = tmp_path / "model"
    run.mkdir()
    target = tmp_path / "maps"
    with pytest.raises(ValueError, match="calibration season"):
        render_map(study_config, day, run, target)
    assert not target.exists()


def test_cli_run_passes_the_validated_plan_to_runner(study_config, monkeypatch):
    plans = []

    def run(plan):
        plans.append(plan)
        return 7

    monkeypatch.setattr(development, "run_plan", run)
    assert development.main(["run", "--config", str(study_config), "--tag", "cli_probe",
                             "--suite", "ablations"]) == 7
    assert len(plans) == 1
    assert len(plans[0].steps) == 7


def test_cli_invalid_options_return_a_usage_error_without_launch(study_config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid plan must not launch")

    monkeypatch.setattr(development, "run_plan", forbidden)
    with pytest.raises(SystemExit) as error:
        development.main(["run", "--config", str(study_config), "--tag", "../bad"])
    assert error.value.code == 2
