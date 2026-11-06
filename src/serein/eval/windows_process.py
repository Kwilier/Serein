"""Windows child jobs for the development runner, without process enumeration."""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from ctypes import wintypes as wt


class _StartupInfo(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("reserved", wt.LPWSTR), ("desktop", wt.LPWSTR),
                ("title", wt.LPWSTR), ("x", wt.DWORD), ("y", wt.DWORD),
                ("x_size", wt.DWORD), ("y_size", wt.DWORD), ("x_chars", wt.DWORD),
                ("y_chars", wt.DWORD), ("fill", wt.DWORD), ("flags", wt.DWORD),
                ("show", wt.WORD), ("reserved_size", wt.WORD), ("reserved_bytes", wt.LPBYTE),
                ("stdin", wt.HANDLE), ("stdout", wt.HANDLE), ("stderr", wt.HANDLE)]


class _ProcessInfo(ctypes.Structure):
    _fields_ = [("process", wt.HANDLE), ("thread", wt.HANDLE),
                ("pid", wt.DWORD), ("tid", wt.DWORD)]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("startup", _StartupInfo), ("attributes", ctypes.c_void_p)]


class _BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", wt.DWORD), ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t), ("active_processes", wt.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wt.DWORD), ("scheduling", wt.DWORD)]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", _BasicLimits), ("io_counters", ctypes.c_uint64 * 6),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [("total_user_time", ctypes.c_int64), ("total_kernel_time", ctypes.c_int64),
                ("period_user_time", ctypes.c_int64), ("period_kernel_time", ctypes.c_int64),
                ("page_faults", wt.DWORD), ("total_processes", wt.DWORD),
                ("active_processes", wt.DWORD), ("terminated_processes", wt.DWORD)]


def _api():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, wt.LPCWSTR], wt.HANDLE),
        "SetInformationJobObject": ([wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD], wt.BOOL),
        "TerminateJobObject": ([wt.HANDLE, wt.UINT], wt.BOOL),
        "TerminateProcess": ([wt.HANDLE, wt.UINT], wt.BOOL),
        "QueryInformationJobObject": ([wt.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                        wt.DWORD, ctypes.c_void_p], wt.BOOL),
        "CloseHandle": ([wt.HANDLE], wt.BOOL),
        "ResumeThread": ([wt.HANDLE], wt.DWORD),
        "WaitForSingleObject": ([wt.HANDLE, wt.DWORD], wt.DWORD),
        "GetExitCodeProcess": ([wt.HANDLE, ctypes.POINTER(wt.DWORD)], wt.BOOL),
        "InitializeProcThreadAttributeList": ([ctypes.c_void_p, wt.DWORD, wt.DWORD,
                                               ctypes.POINTER(ctypes.c_size_t)], wt.BOOL),
        "UpdateProcThreadAttribute": ([ctypes.c_void_p, wt.DWORD, ctypes.c_size_t,
                                       ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.c_void_p, ctypes.c_void_p], wt.BOOL),
        "DeleteProcThreadAttributeList": ([ctypes.c_void_p], None),
        "CreateProcessW": ([wt.LPCWSTR, wt.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
                            wt.BOOL, wt.DWORD, ctypes.c_void_p, wt.LPCWSTR,
                            ctypes.c_void_p, ctypes.POINTER(_ProcessInfo)], wt.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes, function.restype = arguments, result
    return kernel


def _check(success):
    if not success:
        raise ctypes.WinError(ctypes.get_last_error())


def _stop_job(kernel, job):
    _check(kernel.TerminateJobObject(job, 125))
    deadline = time.monotonic() + 5
    accounting = _Accounting()
    while True:
        _check(kernel.QueryInformationJobObject(job, 1, ctypes.byref(accounting),
                                                ctypes.sizeof(accounting), None))
        if not accounting.active_processes:
            return
        if time.monotonic() >= deadline:
            raise OSError("child job still has active processes after termination")
        time.sleep(.01)


def run_windows(command, *, cwd, env, stdout, timeout=None) -> subprocess.CompletedProcess:
    """Create a suspended process inside a private job before it can create children.

    Job handles are never inherited. Closing the job kills its whole tree, even
    if the Python launcher has exited. No PIDs outside this job are opened.
    """
    import msvcrt

    kernel = _api()
    job = kernel.CreateJobObjectW(None, None)
    _check(job)
    process = _ProcessInfo()
    descriptors = []
    assigned = False
    attributes = None
    attributes_initialized = False
    try:
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        _check(kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        # Duplicate only our log and NUL input; do not change caller-owned handles.
        with open(os.devnull, "rb") as null:
            for descriptor in (null.fileno(), stdout.fileno()):
                duplicate = os.dup(descriptor)
                descriptors.append(duplicate)
                os.set_inheritable(duplicate, True)
        startup = _StartupInfoEx()
        startup.startup.cb = ctypes.sizeof(startup)
        startup.startup.flags = 0x100  # STARTF_USESTDHANDLES
        startup.startup.stdin = msvcrt.get_osfhandle(descriptors[0])
        startup.startup.stdout = startup.startup.stderr = msvcrt.get_osfhandle(descriptors[1])
        size = ctypes.c_size_t()
        kernel.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
        if not size.value:
            raise ctypes.WinError(ctypes.get_last_error())
        attributes = ctypes.create_string_buffer(size.value)
        _check(kernel.InitializeProcThreadAttributeList(attributes, 2, 0, ctypes.byref(size)))
        attributes_initialized = True
        handles = (wt.HANDLE * 2)(startup.startup.stdin, startup.startup.stdout)
        # PROC_THREAD_ATTRIBUTE_HANDLE_LIST excludes unrelated inheritable handles.
        _check(kernel.UpdateProcThreadAttribute(attributes, 0, 0x20002, handles,
                                                ctypes.sizeof(handles), None, None))
        jobs = (wt.HANDLE * 1)(job)
        # PROC_THREAD_ATTRIBUTE_JOB_LIST assigns ownership atomically at creation.
        _check(kernel.UpdateProcThreadAttribute(attributes, 0, 0x2000D, jobs,
                                                ctypes.sizeof(jobs), None, None))
        startup.attributes = ctypes.cast(attributes, ctypes.c_void_p)
        environment = ctypes.create_unicode_buffer(
            "\0".join(f"{key}={value}" for key, value in sorted(env.items(), key=lambda x: x[0].upper()))
            + "\0\0")
        line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
        # CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW |
        # EXTENDED_STARTUPINFO_PRESENT
        flags = 0x4 | 0x400 | 0x08000000 | 0x80000
        _check(kernel.CreateProcessW(str(command[0]), line, None, None, True, flags,
                                     environment, str(cwd), ctypes.byref(startup), ctypes.byref(process)))
        assigned = True
        for descriptor in descriptors:
            os.close(descriptor)
        descriptors.clear()
        print(f"process pid: {process.pid}", file=stdout, flush=True)
        if kernel.ResumeThread(process.thread) == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            # Bounded waits let Python deliver KeyboardInterrupt promptly.
            wait_ms = 100 if remaining is None else min(100, max(1, int(remaining * 1000)))
            status = kernel.WaitForSingleObject(process.process, wait_ms)
            if status == 0:
                result = wt.DWORD()
                _check(kernel.GetExitCodeProcess(process.process, ctypes.byref(result)))
                return subprocess.CompletedProcess(command, result.value)
            if status != 0x102:  # WAIT_TIMEOUT
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        try:
            if process.process:
                if assigned:
                    _stop_job(kernel, job)
                else:
                    _check(kernel.TerminateProcess(process.process, 125))
                if kernel.WaitForSingleObject(process.process, 5000) == 0x102:
                    raise OSError("child process did not stop after job termination")
        finally:
            if attributes_initialized:
                kernel.DeleteProcThreadAttributeList(attributes)
            for descriptor in descriptors:
                os.close(descriptor)
            for handle in (process.thread, process.process, job):
                if handle:
                    kernel.CloseHandle(handle)
