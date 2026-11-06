import json
import sys

import pytest

from serein.eval import runtime_check
from serein.eval.runtime_check import check_runtime, main


def worker(monkeypatch, stages=None, *, exit_code=0, sleep=0):
    if stages is None:
        stages = ["import_torch", "torch_imported", "allocate", "forward", "backward", "complete"]
    code = ("import json,sys,time; "
            f"records={stages!r}; "
            "[print(json.dumps({'stage':stage}),flush=True) for stage in records]; "
            f"time.sleep({sleep}); sys.exit({exit_code})")
    monkeypatch.setattr(runtime_check, "_command", lambda device: [sys.executable, "-c", code])


def test_real_child_checkpoints_log_and_manifest_are_retained(tmp_path, monkeypatch):
    worker(monkeypatch)
    output = tmp_path / "runtime"
    assert check_runtime(output) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "passed"
    assert manifest["last_stage"] == "complete"
    assert manifest["exit_code"] == 0
    assert manifest["device"] == "cpu"
    assert "process pid:" in (output / "probe.log").read_text()


def test_successful_exit_without_backward_check_is_a_failure(tmp_path, monkeypatch):
    worker(monkeypatch, stages=["import_torch", "complete"])
    output = tmp_path / "runtime"
    assert check_runtime(output) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "checkpoints" in manifest["error"]


def test_child_failure_preserves_exit_code_and_last_stage(tmp_path, monkeypatch):
    worker(monkeypatch, stages=["import_torch"], exit_code=7)
    output = tmp_path / "runtime"
    assert check_runtime(output) == 7
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["last_stage"] == "import_torch"


def test_hung_probe_times_out_and_keeps_progress(tmp_path, monkeypatch):
    worker(monkeypatch, stages=["import_torch", "torch_imported", "forward", "backward"], sleep=2)
    output = tmp_path / "runtime"
    assert check_runtime(output, timeout_seconds=.5) == 124
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "timed_out"
    assert manifest["last_stage"] == "backward"


def test_existing_outputs_are_never_overwritten(tmp_path, monkeypatch):
    worker(monkeypatch)
    output = tmp_path / "runtime"
    output.mkdir()
    original = output / "probe.log"
    original.write_text("keep this")
    with pytest.raises(FileExistsError):
        check_runtime(output)
    assert original.read_text() == "keep this"
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("options", [
    {"device": "other"}, {"timeout_seconds": 0}, {"timeout_seconds": -1},
    {"timeout_seconds": True}, {"timeout_seconds": float("inf")}, {"timeout_seconds": float("nan")},
])
def test_invalid_probe_options_do_not_create_outputs(tmp_path, options):
    output = tmp_path / "runtime"
    with pytest.raises(ValueError):
        check_runtime(output, **options)
    assert not output.exists()


def test_launch_failure_is_reported_with_its_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_check, "_command", lambda device: [str(tmp_path / "missing.exe")])
    output = tmp_path / "runtime"
    assert check_runtime(output) == 127
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]
    assert manifest["last_stage"] is None


def test_cli_requires_an_explicit_new_output_directory(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        main([])
    worker(monkeypatch)
    assert main(["--output", str(tmp_path / "cli")]) == 0


def test_interruption_retains_a_failed_manifest(tmp_path, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime_check, "run_logged", interrupt)
    output = tmp_path / "runtime"
    assert check_runtime(output) == 130
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["exit_code"] == 130


def test_unexpected_error_cannot_leave_a_running_or_passed_manifest(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected probe failure")

    monkeypatch.setattr(runtime_check, "run_logged", fail)
    output = tmp_path / "runtime"
    with pytest.raises(RuntimeError, match="unexpected probe failure"):
        check_runtime(output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == 1
