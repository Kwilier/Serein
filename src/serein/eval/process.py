"""Run logged development commands with ownership of their process trees."""

from __future__ import annotations

import os
import signal
import subprocess


def run_logged(command, *, cwd, env, stdout, timeout=None) -> subprocess.CompletedProcess:
    """Keep descendants inside a step's lifetime, including on timeout or Ctrl-C."""
    if os.name == "nt":
        from serein.eval.windows_process import run_windows

        return run_windows(command, cwd=cwd, env=env, stdout=stdout, timeout=timeout)
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True)
    try:
        print(f"process pid: {process.pid}", file=stdout, flush=True)
        return subprocess.CompletedProcess(command, process.wait(timeout=timeout))
    finally:
        # A completed parent can still have workers holding files or consuming CPU.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
