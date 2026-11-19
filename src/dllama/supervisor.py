"""Owns the `ollama serve` child process: start/stop/restart, log capture, crash restarts."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any
from collections.abc import Callable

import aiohttp
import psutil

from .config import ollama_env

BACKOFF = (1, 2, 5, 10)  # seconds between crash restarts
LOG_LINES = 1000


def find_binary(configured: str) -> str:
    if configured:
        return configured
    found = shutil.which("ollama")
    if found:
        return found
    default = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if default.exists():
        return str(default)
    raise FileNotFoundError("ollama executable not found; set [ollama].binary in the config")


class _Job:
    """Windows Job Object that kills every process in it when closed, including when dllama itself dies."""

    def __init__(self) -> None:
        import win32job

        self._job: Any = win32job.CreateJobObject(None, "")  # PyHANDLE; the stubs mistype it
        info = win32job.QueryInformationJobObject(self._job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(self._job, win32job.JobObjectExtendedLimitInformation, info)

    def add(self, pid: int) -> None:
        import win32api
        import win32con
        import win32job

        handle = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, pid)
        try:
            win32job.AssignProcessToJobObject(self._job, handle)
        finally:
            win32api.CloseHandle(handle)

    def close(self) -> None:
        self._job.Close()


class Supervisor:
    def __init__(
        self,
        cfg: dict,
        session: aiohttp.ClientSession,
        on_status: Callable[[dict], None],
        on_log: Callable[[str], None],
    ) -> None:
        self.cfg = cfg
        self.session = session
        self.on_status = on_status
        self.on_log = on_log
        self.logs: deque[str] = deque(maxlen=LOG_LINES)
        self.status = "stopped"
        self.version: str | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._job: _Job | None = None
        self._reader: asyncio.Task | None = None
        self._task: asyncio.Task | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.cfg['ollama']['internal_port']}"

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def active(self) -> bool:
        """True while dllama wants ollama running (including starting/restarting)."""
        return self._task is not None

    def state(self) -> dict:
        return {"status": self.status, "version": self.version, "pid": self.pid}

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    def _set_status(self, status: str) -> None:
        self.status = status
        self.on_status(self.state())

    def _log(self, line: str) -> None:
        self.logs.append(line)
        self.on_log(line)

    async def _run(self) -> None:
        failures = 0
        try:
            while True:
                self._set_status("restarting" if failures else "starting")
                started = time.monotonic()
                try:
                    code = await self._run_once()
                    self._log(f"[dllama] ollama exited with code {code}")
                except OSError as e:
                    self._log(f"[dllama] failed to start ollama: {e}")
                await self._kill()
                self.version = None
                failures = 1 if time.monotonic() - started > 60 else failures + 1
                delay = BACKOFF[min(failures, len(BACKOFF)) - 1]
                self._set_status("restarting")
                self._log(f"[dllama] restarting in {delay}s")
                await asyncio.sleep(delay)
        finally:
            await self._kill()
            self.version = None
            self._set_status("stopped")
            self._log("[dllama] ollama stopped")

    async def _run_once(self) -> int:
        binary = find_binary(self.cfg["ollama"]["binary"])
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0  # keep Ctrl+C away from ollama
        self._proc = await asyncio.create_subprocess_exec(
            binary,
            "serve",
            env=ollama_env(dict(os.environ), self.cfg),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=flags,
        )
        if sys.platform == "win32":
            self._job = _Job()
            self._job.add(self._proc.pid)
        assert self._proc.stdout is not None
        self._reader = asyncio.create_task(self._read_logs(self._proc.stdout))
        self._log(f"[dllama] started {binary} serve (pid {self._proc.pid})")
        ready = asyncio.create_task(self._wait_ready())
        try:
            # Poll returncode instead of awaiting wait(): wait() only resolves once every pipe closes,
            # and orphaned runner processes can hold the stdout pipe open.
            while self._proc.returncode is None:
                await asyncio.sleep(0.5)
            await asyncio.wait({self._reader}, timeout=2)  # let the last log lines arrive first
            return self._proc.returncode
        finally:
            ready.cancel()

    async def _read_logs(self, stream: asyncio.StreamReader) -> None:
        buf = b""
        while chunk := await stream.read(65536):
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                self._log(line.decode(errors="replace").rstrip())
        if buf:
            self._log(buf.decode(errors="replace").rstrip())

    async def _wait_ready(self) -> None:
        timeout = aiohttp.ClientTimeout(total=2)
        for _ in range(120):
            try:
                async with self.session.get(f"{self.base_url}/api/version", timeout=timeout) as resp:
                    if resp.status == 200:
                        self.version = (await resp.json()).get("version")
                        self._set_status("running")
                        return
            except (TimeoutError, aiohttp.ClientError, ValueError):
                pass
            await asyncio.sleep(0.5)
        self._log("[dllama] ollama did not answer within 60s")

    async def _kill(self) -> None:
        """Kill `ollama serve` and all its runners, which frees the VRAM."""
        proc, self._proc = self._proc, None
        job, self._job = self._job, None
        reader, self._reader = self._reader, None
        if proc is None:
            return
        try:
            root = psutil.Process(proc.pid)
            procs = [*root.children(recursive=True), root]
        except psutil.Error:
            procs = []
        for p in procs:
            with contextlib.suppress(psutil.Error):
                p.kill()
        if job:
            job.close()  # also catches runners orphaned by a crashed `ollama serve`
        await asyncio.to_thread(psutil.wait_procs, procs, timeout=5)
        if reader:
            await asyncio.wait({reader}, timeout=2)
            reader.cancel()
        with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
            await asyncio.wait_for(proc.wait(), 5)
