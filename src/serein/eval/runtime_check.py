"""Check a small PyTorch forward/backward pass in a bounded, owned subprocess."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from serein.eval.process import run_logged


def _checkpoint(stage: str, **values):
    print(json.dumps({"stage": stage, **values}), flush=True)


def probe(device: str) -> None:
    """Use synthetic tensors only; this does not load data or model artifacts."""
    _checkpoint("import_torch")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _checkpoint("torch_imported", version=torch.__version__, python=sys.version.split()[0],
                device=device, cuda_build=torch.version.cuda,
                num_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
    _checkpoint("allocate")
    torch.manual_seed(17)
    x = torch.randn(4, 16, device=device, requires_grad=True)
    layer = torch.nn.Linear(16, 8).to(device)
    _checkpoint("forward")
    loss = layer(x).square().mean()
    _checkpoint("backward")
    gradient, = torch.autograd.grad(loss, x)
    if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
        raise RuntimeError("the synthetic backward pass did not produce finite nonzero gradients")
    _checkpoint("complete")


def _command(device: str) -> list[str]:
    return [sys.executable, "-m", "serein.eval.runtime_check", "--worker", "--device", device]


def check_runtime(output: Path, *, device: str = "cpu", timeout_seconds: float = 30) -> int:
    """Write an isolated probe log and status, returning 124 for a timed-out probe."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if (isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    output.mkdir(exist_ok=False)
    command = _command(device)
    manifest = {"device": device, "timeout_seconds": timeout_seconds, "command": command,
                "status": "running", "exit_code": None, "checkpoints": []}
    exit_code = 1
    try:
        with (output / "probe.log").open("x", encoding="utf-8", newline="") as stream:
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            completed = run_logged(command, cwd=Path.cwd(), env=env, stdout=stream,
                                   timeout=timeout_seconds)
            exit_code = completed.returncode
        manifest["status"] = "passed" if exit_code == 0 else "failed"
    except subprocess.TimeoutExpired:
        exit_code = 124
        manifest["status"] = "timed_out"
    except OSError as exc:
        exit_code = 127
        manifest.update(status="failed", error=str(exc))
    except KeyboardInterrupt:
        exit_code = 130
        manifest["status"] = "interrupted"
    except Exception as exc:
        exit_code = 1
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        log = output / "probe.log"
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and isinstance(record.get("stage"), str):
                manifest["checkpoints"].append(record)
        stages = [record["stage"] for record in manifest["checkpoints"]]
        if manifest["status"] == "passed" and stages != [
            "import_torch", "torch_imported", "allocate", "forward", "backward", "complete",
        ]:
            exit_code = 1
            manifest.update(status="failed", error="probe exited without completing its checkpoints")
        manifest["exit_code"] = exit_code
        manifest["last_stage"] = stages[-1] if stages else None
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"runtime check: {manifest['status']}; last stage {manifest['last_stage']}; log {output / 'probe.log'}")
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory for the log and manifest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        probe(args.device)
        return 0
    if args.output is None:
        parser.error("--output is required")
    try:
        return check_runtime(args.output, device=args.device, timeout_seconds=args.timeout_seconds)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
