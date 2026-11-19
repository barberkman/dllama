"""Polls ollama's /api/ps for the loaded models."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from collections.abc import Callable

import aiohttp

POLL_SECONDS = 2
_FRACTION = re.compile(r"\.(\d+)")


def _expires_in(expires_at: str | None) -> float | None:
    """Seconds until unload. Go timestamps carry up to 9 fractional digits; Python takes 6."""
    if not expires_at:
        return None
    text = _FRACTION.sub(lambda m: "." + (m.group(1) + "000000")[:6], expires_at.replace("Z", "+00:00"), count=1)
    try:
        return datetime.fromisoformat(text).timestamp() - time.time()
    except ValueError:
        return None


def parse_ps(data: dict) -> list[dict]:
    models = []
    for m in data.get("models") or []:
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        models.append(
            {
                "name": m.get("name") or m.get("model"),
                "size": size,
                "size_vram": vram,
                "gpu_pct": round(100 * vram / size) if size else 0,
                "context": m.get("context_length"),
                "expires_at": m.get("expires_at"),
                "expires_in": _expires_in(m.get("expires_at")),
            }
        )
    return models


def _key(models: list[dict]) -> list[tuple]:
    return [(m["name"], m["size_vram"], m["context"], m["expires_at"]) for m in models]


class ModelPoller:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        is_running: Callable[[], bool],
        on_models: Callable[[list[dict]], None],
    ) -> None:
        self.session = session
        self.base_url = base_url
        self.is_running = is_running
        self.on_models = on_models
        self.models: list[dict] = []

    async def run(self) -> None:
        while True:
            models: list[dict] = []
            if self.is_running():
                try:
                    timeout = aiohttp.ClientTimeout(total=3)
                    async with self.session.get(f"{self.base_url}/api/ps", timeout=timeout) as resp:
                        models = parse_ps(await resp.json())
                except (TimeoutError, aiohttp.ClientError, ValueError):
                    models = self.models
            changed = _key(models) != _key(self.models)
            self.models = models  # always keep the freshest expires_in for snapshots
            if changed:
                self.on_models(models)
            await asyncio.sleep(POLL_SECONDS)
