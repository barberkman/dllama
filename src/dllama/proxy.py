"""Reverse proxy in front of ollama that records every request (model, tokens, timing)."""

from __future__ import annotations

import itertools
import json
import secrets
import time
from collections import deque
from typing import Any
from collections.abc import Callable

import aiohttp
from aiohttp import web
from multidict import CIMultiDict

MAX_RECORDS = 500
MAX_PARSE_BUFFER = 8 * 1024 * 1024
MAX_INFERENCE_BODY = 256 * 1024 * 1024  # base64 images can be large
INFERENCE_PREFIXES = (
    "/api/chat",
    "/api/generate",
    "/api/embed",
    "/v1/chat",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/messages",
    "/v1/responses",
)
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers",
        "transfer-encoding", "upgrade"}
_SKIP_REQUEST = _HOP | {"host", "accept-encoding"}  # no accept-encoding: keep responses parseable
_SKIP_RESPONSE = _HOP | {"content-length"}  # the response is re-chunked


def is_inference(path: str) -> bool:
    return path.startswith(INFERENCE_PREFIXES)


def extract_usage(obj: Any, usage: dict) -> None:
    """Merge token counts from one response object (ollama native, OpenAI or Anthropic format) into `usage`."""
    if not isinstance(obj, dict):
        return
    for src, dst in (("prompt_eval_count", "prompt_tok"), ("eval_count", "out_tok"), ("eval_duration", "eval_ns")):
        if isinstance(obj.get(src), int):
            usage[dst] = obj[src]
    found = obj.get("usage")
    if not isinstance(found, dict) and isinstance(obj.get("message"), dict):
        found = obj["message"].get("usage")  # Anthropic message_start
    if isinstance(found, dict):
        for src, dst in (("prompt_tokens", "prompt_tok"), ("completion_tokens", "out_tok"),
                         ("input_tokens", "prompt_tok"), ("output_tokens", "out_tok")):
            if isinstance(found.get(src), int):
                usage[dst] = found[src]


class UsageParser:
    """Finds token counts in a JSON, NDJSON or SSE body that arrives in arbitrary chunks."""

    def __init__(self) -> None:
        self.usage: dict = {}
        self._buf = b""

    def feed(self, data: bytes) -> None:
        self._buf += data
        *lines, self._buf = self._buf.split(b"\n")
        for line in lines:
            self._line(line)
        if len(self._buf) > MAX_PARSE_BUFFER:
            self._buf = b""

    def close(self) -> None:
        self._line(self._buf)
        self._buf = b""

    def _line(self, line: bytes) -> None:
        line = line.strip()
        if line.startswith(b"data:"):
            line = line[5:].strip()
        if not line.startswith(b"{"):
            return
        try:
            extract_usage(json.loads(line), self.usage)
        except ValueError:
            pass


def _model(body: bytes) -> str | None:
    if not body.lstrip().startswith(b"{"):
        return None
    try:
        model = json.loads(body).get("model")
    except ValueError:
        return None
    return model if isinstance(model, str) else None


def _tok_s(rec: dict, usage: dict) -> float | None:
    out = usage.get("out_tok")
    if not out:
        return None
    if usage.get("eval_ns"):
        return round(out / (usage["eval_ns"] / 1e9), 1)
    gen_ms = rec["duration_ms"] - (rec["ttft_ms"] or 0)
    return round(out / (gen_ms / 1000), 1) if gen_ms > 50 else None


class Proxy:
    def __init__(self, upstream: str, api_key: str, on_request: Callable[[dict], None]) -> None:
        self.upstream = upstream.rstrip("/")
        self.api_key = api_key
        self.on_request = on_request
        self.records: deque[dict] = deque(maxlen=MAX_RECORDS)
        self._ids = itertools.count(1)
        self._session: aiohttp.ClientSession | None = None

    def app(self) -> web.Application:
        app = web.Application(client_max_size=MAX_INFERENCE_BODY)
        app.router.add_route("*", "/{tail:.*}", self._handle)
        app.on_startup.append(self._open)
        app.on_cleanup.append(self._close)
        return app

    async def _open(self, _app: web.Application) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),  # generations can take minutes
            auto_decompress=False,
            skip_auto_headers=("Accept-Encoding", "Content-Type", "User-Agent"),
        )

    async def _close(self, _app: web.Application) -> None:
        if self._session:
            await self._session.close()

    def _authorized(self, request: web.Request) -> bool:
        if not self.api_key:
            return True
        bearer = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        given = request.headers.get("x-api-key") or bearer
        return secrets.compare_digest(given.encode(), self.api_key.encode())

    def _emit(self, rec: dict) -> None:
        self.on_request(dict(rec))

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        inference = is_inference(request.path)
        if inference:
            body = await request.read()
            data: Any = body or None
            model = _model(body)
        else:  # stream everything else (e.g. multi-GB blob uploads) without buffering
            data = request.content if request.body_exists else None
            model = None
        rec = {
            "id": next(self._ids), "ts": time.time(), "client": request.remote, "method": request.method,
            "path": request.path, "inference": inference, "model": model, "status": None, "state": "active",
            "ttft_ms": None, "duration_ms": None, "prompt_tok": None, "out_tok": None, "tok_s": None,
        }
        self.records.append(rec)
        self._emit(rec)

        parser = UsageParser() if inference else None
        headers = CIMultiDict((k, v) for k, v in request.headers.items() if k.lower() not in _SKIP_REQUEST)
        start = time.perf_counter()
        response: web.StreamResponse | None = None
        assert self._session is not None
        try:
            async with self._session.request(
                request.method, self.upstream + request.path_qs, headers=headers, data=data, allow_redirects=False
            ) as upstream:
                rec["status"] = upstream.status
                response = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=CIMultiDict((k, v) for k, v in upstream.headers.items() if k.lower() not in _SKIP_RESPONSE),
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    if rec["ttft_ms"] is None:
                        rec["ttft_ms"] = round((time.perf_counter() - start) * 1000)
                    if parser:
                        parser.feed(chunk)
                    try:
                        await response.write(chunk)
                    except ConnectionResetError:
                        upstream.close()  # client is gone: drop the upstream connection so ollama stops generating
                        raise
                await response.write_eof()
            rec["state"] = "done" if upstream.status < 400 else "error"
        except ConnectionResetError:
            rec["state"] = "cancelled"
        except aiohttp.ClientError:
            rec["state"] = "error"
            if response is None:
                rec["status"] = 502
                return web.json_response({"error": "ollama is not reachable"}, status=502)
        finally:
            rec["duration_ms"] = round((time.perf_counter() - start) * 1000)
            if parser:
                parser.close()
                rec["prompt_tok"] = parser.usage.get("prompt_tok")
                rec["out_tok"] = parser.usage.get("out_tok")
                rec["tok_s"] = _tok_s(rec, parser.usage)
            if rec["state"] == "active":
                rec["state"] = "cancelled"  # e.g. server shutdown
            self._emit(rec)
        assert response is not None
        return response
