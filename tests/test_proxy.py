import json

import pytest
from aiohttp import web

from dllama.proxy import Proxy

CHAT_BODY = b"".join(
    json.dumps(o).encode() + b"\n"
    for o in (
        {"message": {"role": "assistant", "content": "hel"}, "done": False},
        {"message": {"role": "assistant", "content": "lo"}, "done": False},
        {"done": True, "prompt_eval_count": 12, "eval_count": 34, "eval_duration": 2_000_000_000},
    )
)


@pytest.fixture
async def upstream_url(aiohttp_server):
    async def chat(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
        await resp.prepare(request)
        for line in CHAT_BODY.splitlines(keepends=True):
            await resp.write(line)
        return resp

    async def tags(_: web.Request) -> web.Response:
        return web.json_response({"models": []})

    app = web.Application()
    app.router.add_post("/api/chat", chat)
    app.router.add_get("/api/tags", tags)
    server = await aiohttp_server(app)
    return f"http://{server.host}:{server.port}"


async def make_proxy(aiohttp_client, upstream: str, api_key: str = ""):
    events: list[dict] = []
    proxy = Proxy(upstream, api_key, events.append)
    client = await aiohttp_client(proxy.app())
    return client, proxy, events


async def test_streams_bytes_and_records_usage(aiohttp_client, upstream_url):
    client, proxy, events = await make_proxy(aiohttp_client, upstream_url)
    resp = await client.post("/api/chat", json={"model": "qwen3:14b", "messages": []})
    assert resp.status == 200
    assert await resp.read() == CHAT_BODY

    rec = proxy.records[-1]
    assert (rec["model"], rec["status"], rec["state"]) == ("qwen3:14b", 200, "done")
    assert (rec["prompt_tok"], rec["out_tok"], rec["tok_s"]) == (12, 34, 17.0)
    assert rec["ttft_ms"] is not None and rec["duration_ms"] is not None
    assert [e["state"] for e in events] == ["active", "done"]


async def test_non_inference_request_is_recorded_without_parsing(aiohttp_client, upstream_url):
    client, proxy, _ = await make_proxy(aiohttp_client, upstream_url)
    resp = await client.get("/api/tags")
    assert await resp.json() == {"models": []}
    rec = proxy.records[-1]
    assert (rec["inference"], rec["model"], rec["out_tok"], rec["state"]) == (False, None, None, "done")


async def test_api_key_is_enforced(aiohttp_client, upstream_url):
    client, proxy, _ = await make_proxy(aiohttp_client, upstream_url, api_key="secret")
    assert (await client.get("/api/tags")).status == 401
    assert (await client.get("/api/tags", headers={"Authorization": "Bearer wrong"})).status == 401
    assert (await client.get("/api/tags", headers={"Authorization": "Bearer secret"})).status == 200
    assert (await client.get("/api/tags", headers={"x-api-key": "secret"})).status == 200
    assert len(proxy.records) == 2  # rejected requests never reach ollama


async def test_upstream_down_returns_502(aiohttp_client, unused_tcp_port):
    client, proxy, _ = await make_proxy(aiohttp_client, f"http://127.0.0.1:{unused_tcp_port}")
    resp = await client.post("/api/chat", json={"model": "m"})
    assert resp.status == 502
    assert (proxy.records[-1]["status"], proxy.records[-1]["state"]) == (502, "error")
