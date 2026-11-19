"""1 Hz sampler for system, ollama-process and GPU usage, with a fixed-size history."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable

import psutil

from .gpu import Gpu

HISTORY = 300  # samples (seconds)


class Metrics:
    def __init__(self, gpu: Gpu, get_pid: Callable[[], int | None], on_sample: Callable[[dict], None]) -> None:
        self.gpu = gpu
        self.get_pid = get_pid
        self.on_sample = on_sample
        self.history: deque[dict] = deque(maxlen=HISTORY)
        self._procs: dict[int, psutil.Process] = {}
        self._ncpu = psutil.cpu_count() or 1

    async def run(self) -> None:
        psutil.cpu_percent(None)  # the first call only sets the baseline
        while True:
            await asyncio.sleep(1)
            sample = await asyncio.to_thread(self._sample)
            self.history.append(sample)
            self.on_sample(sample)

    def _sample(self) -> dict:
        vm = psutil.virtual_memory()
        ollama_cpu, ollama_rss = self._ollama_usage()
        return {
            "ts": time.time(),
            "cpu": psutil.cpu_percent(None),
            "ram_used": vm.total - vm.available,
            "ram_total": vm.total,
            "ollama_cpu": ollama_cpu,
            "ollama_rss": ollama_rss,
            **self.gpu.sample(),
        }

    def _ollama_usage(self) -> tuple[float, int]:
        """CPU % of the whole machine and RSS, summed over `ollama serve` and its runner processes."""
        pid = self.get_pid()
        procs: dict[int, psutil.Process] = {}
        if pid is not None:
            try:
                root = psutil.Process(pid)
                for p in [root, *root.children(recursive=True)]:
                    procs[p.pid] = self._procs.get(p.pid, p)  # reuse objects so cpu_percent has a baseline
            except psutil.Error:
                pass
        cpu, rss = 0.0, 0
        for p in procs.values():
            try:
                cpu += p.cpu_percent(None)
                rss += p.memory_info().rss
            except psutil.Error:
                pass
        self._procs = procs
        return cpu / self._ncpu, rss
