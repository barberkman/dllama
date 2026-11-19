"""`dllama serve` / `dllama up`: runs ollama, the API proxy and the control API that the dashboard connects to."""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import socket
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp
import psutil
from aiohttp import web

from . import config
from .gpu import Gpu
from .metrics import Metrics
from .ollama_poll import ModelPoller
from .proxy import Proxy
from .supervisor import Supervisor

TRAY_HINT = "The Ollama tray app is running: quit it (tray icon > Quit Ollama) and turn off its start-at-login."


class Hub:
    """Fans events out to every connected dashboard websocket."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[str]] = set()

    def publish(self, kind: str, data: Any) -> None:
        msg = json.dumps({"type": kind, "data": data})
        for q in self._queues:
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(msg)

    async def serve_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=5000)
        self._queues.add(queue)

        async def pump() -> None:
            with contextlib.suppress(ConnectionResetError):
                while True:
                    await ws.send_str(await queue.get())

        sender = asyncio.create_task(pump())
        try:
            async for _ in ws:  # the dashboard never sends; this just waits for it to go away
                pass
        finally:
            self._queues.discard(queue)
            sender.cancel()
        return ws


def control_app(
    cfg: dict, cfg_path: Path, sup: Supervisor, proxy: Proxy, metrics: Metrics, poller: ModelPoller,
    hub: Hub, gpu: Gpu,
) -> web.Application:
    expected = f"Bearer {cfg['control']['token']}".encode()

    @web.middleware
    async def auth(request: web.Request, handler):
        if not secrets.compare_digest(request.headers.get("Authorization", "").encode(), expected):
            return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)

    async def snapshot(_: web.Request) -> web.Response:
        return web.json_response({
            **sup.state(),
            "env": cfg["ollama"]["env"],
            "gpu_name": gpu.name,
            "hostname": socket.gethostname(),
            "metrics": list(metrics.history),
            "models": poller.models,
            "requests": list(proxy.records),
            "logs": list(sup.logs),
        })

    async def ollama_action(request: web.Request) -> web.Response:
        actions = {"start": sup.start, "stop": sup.stop, "restart": sup.restart}
        await actions[request.match_info["action"]]()
        return web.json_response(sup.state())

    async def get_config(_: web.Request) -> web.Response:
        return web.json_response({"env": cfg["ollama"]["env"]})

    async def put_config(request: web.Request) -> web.Response:
        try:
            env = (await request.json())["env"]
        except (ValueError, KeyError, TypeError):
            return web.json_response({"error": 'body must be {"env": {...}}'}, status=400)
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            return web.json_response({"error": "env must map strings to strings"}, status=400)
        if any(k.upper() == "OLLAMA_HOST" for k in env):
            return web.json_response({"error": "OLLAMA_HOST is managed by dllama; change [proxy].bind instead"}, status=400)
        cfg["ollama"]["env"] = env
        config.save(cfg_path, cfg)
        hub.publish("config", {"env": env})
        if sup.active:
            await sup.restart()
        return web.json_response({"env": env})

    app = web.Application(middlewares=[auth])
    app.router.add_get("/snapshot", snapshot)
    app.router.add_get("/stream", hub.serve_ws)
    app.router.add_post("/ollama/{action:start|stop|restart}", ollama_action)
    app.router.add_get("/config", get_config)
    app.router.add_put("/config", put_config)
    return app


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def preflight(cfg: dict) -> None:
    ports = {
        "ollama API proxy": config.split_bind(cfg["proxy"]["bind"])[1],
        "dashboard": config.split_bind(cfg["control"]["bind"])[1],
        "ollama (internal)": cfg["ollama"]["internal_port"],
    }
    busy = [f"  port {port} ({name})" for name, port in ports.items() if _listening(port)]
    tray = any((p.info["name"] or "").lower() == "ollama app.exe" for p in psutil.process_iter(["name"]))
    if busy:
        sys.exit("These ports are already in use:\n" + "\n".join(busy) + (f"\n{TRAY_HINT}" if tray else ""))
    if tray:
        print(f"warning: {TRAY_HINT}", file=sys.stderr)


def banner(cfg: dict, cfg_path: Path, created: bool) -> None:
    ips = [
        a.address
        for addrs in psutil.net_if_addrs().values()
        for a in addrs
        if a.family == socket.AF_INET and not a.address.startswith("127.")
    ]
    proxy_port = config.split_bind(cfg["proxy"]["bind"])[1]
    control_port = config.split_bind(cfg["control"]["bind"])[1]
    print(f"dllama serve   config: {cfg_path}{'  (created)' if created else ''}")
    print(f"  addresses:     {', '.join(ips) or 'none found'}")
    print(f"  ollama API:    port {proxy_port}  (point LLM clients at http://<address>:{proxy_port})")
    print(f"  dashboard:     port {control_port}  (dllama connect <address> --token <token>)")
    print(f"  token:         {cfg['control']['token']}")
    if created:
        print(
            "  firewall:      New-NetFirewallRule -DisplayName dllama -Direction Inbound -Protocol TCP "
            f"-LocalPort {proxy_port},{control_port} -RemoteAddress LocalSubnet -Action Allow"
        )


async def serve(
    cfg: dict, cfg_path: Path, stop: asyncio.Event | None = None, on_ready: Callable[[], None] | None = None,
) -> None:
    """Serve until `stop` is set (forever without one). `on_ready` is called once both ports are bound."""
    hub = Hub()
    gpu = Gpu()
    async with aiohttp.ClientSession() as session:
        sup = Supervisor(cfg, session, on_status=lambda s: hub.publish("status", s),
                         on_log=lambda line: hub.publish("log", line))
        proxy = Proxy(sup.base_url, cfg["proxy"]["api_key"], on_request=lambda r: hub.publish("request", r))
        metrics = Metrics(gpu, lambda: sup.pid, on_sample=lambda s: hub.publish("metrics", s))
        poller = ModelPoller(session, sup.base_url, lambda: sup.status == "running",
                             on_models=lambda m: hub.publish("models", m))
        apps = (
            (proxy.app(), cfg["proxy"]["bind"]),
            (control_app(cfg, cfg_path, sup, proxy, metrics, poller, hub, gpu), cfg["control"]["bind"]),
        )
        runners: list[web.AppRunner] = []
        tasks: list[asyncio.Task] = []
        try:
            for app, bind in apps:
                runner = web.AppRunner(app, access_log=None, shutdown_timeout=2)
                await runner.setup()
                runners.append(runner)
                await web.TCPSite(runner, *config.split_bind(bind)).start()
            tasks = [asyncio.create_task(metrics.run()), asyncio.create_task(poller.run())]
            if cfg["ollama"]["autostart"]:
                await sup.start()
            if on_ready:
                on_ready()
            await (stop or asyncio.Event()).wait()
        finally:
            await sup.stop()
            for task in tasks:
                task.cancel()
            for runner in runners:
                await runner.cleanup()


def run(cfg_path: Path) -> None:
    cfg, created = config.load_server(cfg_path)
    preflight(cfg)
    banner(cfg, cfg_path, created)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(cfg, cfg_path))


def serve_in_thread(cfg: dict, cfg_path: Path) -> Callable[[], None]:
    """Start `serve` on its own event loop in a background thread and return a function that stops it.

    Raises whatever stopped the server from starting (for example a port that cannot be bound).
    """
    ready = threading.Event()
    state: dict[str, Any] = {}

    async def main() -> None:
        state["loop"] = asyncio.get_running_loop()
        state["stop"] = asyncio.Event()
        await serve(cfg, cfg_path, state["stop"], ready.set)

    def target() -> None:
        try:
            asyncio.run(main())
        except BaseException as e:  # handed to the starting thread
            state["error"] = e
        finally:
            ready.set()

    # Daemon: if shutdown hangs, the process still exits and the job object takes ollama down with it.
    thread = threading.Thread(target=target, name="dllama-serve", daemon=True)
    thread.start()
    ready.wait()
    if "error" in state:
        raise state["error"]

    def stop() -> None:
        if thread.is_alive():
            state["loop"].call_soon_threadsafe(state["stop"].set)
            thread.join(timeout=15)

    return stop


def run_with_dashboard(cfg_path: Path) -> None:
    """`dllama up`: serve and show the dashboard in this terminal. Quitting the dashboard stops ollama.

    If a `dllama serve` already runs on this machine, only the dashboard is opened.
    """
    from .tui import Dashboard

    cfg, created = config.load_server(cfg_path)
    control = cfg["control"]["bind"]
    host, port = config.local_address(control), config.split_bind(control)[1]
    token = cfg["control"]["token"]
    if _listening(port):
        Dashboard(host, port, token).run()
        return
    preflight(cfg)
    banner(cfg, cfg_path, created)
    try:
        stop = serve_in_thread(cfg, cfg_path)
    except OSError as e:
        sys.exit(f"dllama could not start: {e}")
    try:
        Dashboard(host, port, token, owns_server=True).run()
    finally:
        print("Stopping Ollama...")
        stop()
