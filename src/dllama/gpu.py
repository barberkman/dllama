"""NVIDIA GPU 0 stats via NVML. Every field is None when NVML is unavailable."""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

EMPTY = {"gpu_util": None, "vram_used": None, "vram_total": None, "gpu_temp": None, "gpu_power": None}


class Gpu:
    def __init__(self) -> None:
        self.name: str | None = None
        self._nv: Any = None
        self._handle: Any = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nv = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.name = name.decode() if isinstance(name, bytes) else name
        except Exception:
            self._handle = None

    def _read(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except self._nv.NVMLError:
            return None

    def sample(self) -> dict:
        if self._handle is None:
            return dict(EMPTY)
        nv, h = self._nv, self._handle
        mem = self._read(lambda: nv.nvmlDeviceGetMemoryInfo(h))
        util = self._read(lambda: nv.nvmlDeviceGetUtilizationRates(h))
        power_mw = self._read(lambda: nv.nvmlDeviceGetPowerUsage(h))
        return {
            "gpu_util": util.gpu if util else None,
            "vram_used": mem.used if mem else None,
            "vram_total": mem.total if mem else None,
            "gpu_temp": self._read(lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
            "gpu_power": power_mw / 1000 if power_mw is not None else None,
        }
