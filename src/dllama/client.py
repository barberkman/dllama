"""Dashboard side of the control API: snapshot + websocket events, reconnecting forever."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from collections.abc import Callable

import aiohttp


class Client:
    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        on_snapshot: Callable[[dict], None],
        on_event: Callable[[dict], None],
        on_connection: Callable[[str], None],
    ) -> None:
        self.base = f"http://{host}:{port}"
        self._token = token
        self.on_snapshot = on_snapshot
        self.on_event = on_event
        self.on_connection = on_connection
        self._session: aiohttp.ClientSession | None = None

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=5)
        headers = {"Authorization": f"Bearer {self._token}"}
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            self._session = session
            while True:
                try:
                    # Subscribe first, then load the snapshot, so no event falls in between.
                    async with session.ws_connect(f"{self.base}/stream", heartbeat=20) as ws:
                        async with session.get(f"{self.base}/snapshot", timeout=aiohttp.ClientTimeout(total=15)) as r:
                            r.raise_for_status()
                            self.on_snapshot(await r.json())
                        self.on_connection("connected")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self.on_event(json.loads(msg.data))
                    self.on_connection("disconnected, retrying")
                except aiohttp.ClientResponseError as e:
                    self.on_connection("wrong token" if e.status == 401 else f"server error {e.status}, retrying")
                except (TimeoutError, aiohttp.ClientError, OSError) as e:
                    self.on_connection(f"unreachable ({type(e).__name__}), retrying")
                await asyncio.sleep(3)

    async def request(self, method: str, path: str, payload: Any = None) -> tuple[int, str]:
        if self._session is None:
            return 0, "not connected"
        timeout = aiohttp.ClientTimeout(total=30)
        async with self._session.request(method, self.base + path, json=payload, timeout=timeout) as r:
            return r.status, await r.text()
