"""`dllama connect`: Textual dashboard for a (usually remote) `dllama serve`."""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable

from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Input, Label, RichLog, Static
from textual_plotext import PlotextPlot

from .client import Client

WINDOW = 300  # seconds of history in the charts
GB = 1024**3
MAX_REQUESTS = 500
SHOWN_REQUESTS = 200


def _gb(n: float | None) -> str:
    return f"{n / GB:.1f}G" if n is not None else "–"


def _num(v: float | None, fmt: str = "{}") -> str:
    return fmt.format(v) if v is not None else "–"


def _ms(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v:.0f}ms" if v < 1000 else f"{v / 1000:.1f}s"


def _countdown(expires_in: float | None, elapsed: float) -> str:
    if expires_in is None:
        return "–"
    left = expires_in - elapsed
    if left > 365 * 86400:
        return "never"
    if left <= 0:
        return "now"
    return f"{int(left // 60)}m{int(left % 60):02d}s"


class Chart(PlotextPlot):
    def draw(self, title: str, series: list[tuple[str, list[float], list[float]]], ymax: float) -> None:
        plt = self.plt
        plt.clear_data()
        plt.title(title)
        for label, xs, ys in series:
            if xs:
                plt.plot(xs, ys, marker="braille", label=label)
        plt.xlim(-WINDOW, 0)
        plt.ylim(0, max(ymax, 1))
        self.refresh()


class Confirm(ModalScreen[bool]):
    BINDINGS = [Binding("y", "answer(True)", "Yes"), Binding("n,escape", "answer(False)", "No")]

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.question)
            with Horizontal(classes="buttons"):
                yield Button("Yes (y)", variant="error", id="yes")
                yield Button("No (n)", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, value: bool) -> None:
        self.dismiss(value)


class ConfigScreen(ModalScreen["dict[str, str] | None"]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, env: dict[str, str]) -> None:
        super().__init__()
        self.env = env

    @staticmethod
    def _row(key: str, value: str) -> Horizontal:
        return Horizontal(
            Input(value=key, placeholder="OLLAMA_...", classes="key"),
            Input(value=value, placeholder="value", classes="val"),
            classes="row",
        )

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog wide"):
            yield Label("Ollama environment. Saving restarts Ollama if it is running. Clear a name to remove it.")
            with VerticalScroll(id="rows"):
                for key, value in self.env.items():
                    yield self._row(key, value)
            with Horizontal(classes="buttons"):
                yield Button("Add variable", id="add")
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add":
            self.query_one("#rows").mount(self._row("", ""))
        elif event.button.id == "save":
            env = {}
            for row in self.query(".row"):
                key = row.query_one(".key", Input).value.strip()
                if key:
                    env[key] = row.query_one(".val", Input).value.strip()
            self.dismiss(env)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class Dashboard(App):
    TITLE = "dllama"
    CSS = """
    #status { height: 2; padding: 0 1; background: $boost; }
    #charts { grid-size: 2 2; height: 2fr; min-height: 16; }
    #models { height: auto; max-height: 8; border: round $primary; }
    #requests { height: 1fr; min-height: 6; border: round $primary; }
    #log { height: 12; display: none; border: round $primary; }
    Confirm, ConfigScreen { align: center middle; }
    .dialog { width: 64; height: auto; padding: 1 2; border: thick $primary; background: $surface; }
    .dialog.wide { width: 96; }
    #rows { height: auto; max-height: 20; }
    .row { height: auto; }
    .row .key { width: 3fr; }
    .row .val { width: 2fr; }
    .buttons { height: auto; margin-top: 1; }
    .buttons Button { margin-right: 1; }
    """
    BINDINGS = [
        Binding("s", "toggle_ollama", "Start/Stop"),
        Binding("r", "restart", "Restart"),
        Binding("c", "config", "Config"),
        Binding("l", "toggle_log", "Logs"),
        Binding("f", "toggle_filter", "All requests"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, host: str, port: int, token: str, owns_server: bool = False) -> None:
        super().__init__()
        self.owns_server = owns_server  # `dllama up`: the server lives in this process, so quitting stops ollama
        self.target = f"{host}:{port}"
        self.client = Client(host, port, token, self.on_snapshot, self.on_server_event, self.on_connection)
        self.connection = "connecting"
        self.server: dict = {}
        self.metrics: deque[dict] = deque(maxlen=WINDOW)
        self.models: list[dict] = []
        self.models_at = 0.0
        self.requests: dict[int, dict] = {}
        self.show_all = False
        self._requests_dirty = False

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        with Grid(id="charts"):
            yield Chart(id="cpu")
            yield Chart(id="ram")
            yield Chart(id="gpu")
            yield Chart(id="vram")
        yield DataTable(id="models", cursor_type="none")
        yield DataTable(id="requests", cursor_type="none")
        yield RichLog(id="log", max_lines=1000, markup=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        models = self.query_one("#models", DataTable)
        models.border_title = "Loaded models"
        models.add_columns("Model", "Size", "VRAM", "On GPU", "Context", "Unloads in")
        requests = self.query_one("#requests", DataTable)
        requests.add_columns("Time", "Client", "Endpoint", "Model", "Status", "TTFT", "Duration", "In", "Out", "tok/s")
        self._set_requests_title()
        self.query_one("#log", RichLog).border_title = "Ollama log"
        self.refresh_view()
        self.set_interval(1, self.refresh_view)
        self.run_worker(self.client.run(), exclusive=True)

    # ---- data from the server ----

    def on_connection(self, state: str) -> None:
        self.connection = state
        self._render_status()

    def on_snapshot(self, snap: dict) -> None:
        self.server = {k: snap.get(k) for k in ("status", "version", "pid", "env", "gpu_name", "hostname")}
        self.metrics.clear()
        self.metrics.extend(snap["metrics"])
        self._set_models(snap["models"])
        self.requests = {r["id"]: r for r in snap["requests"]}
        self._requests_dirty = True
        log = self.query_one("#log", RichLog)
        log.clear()
        for line in snap["logs"]:
            log.write(line)
        self.refresh_view()

    def on_server_event(self, event: dict) -> None:
        kind, data = event["type"], event["data"]
        if kind == "metrics":
            if not self.metrics or data["ts"] > self.metrics[-1]["ts"]:
                self.metrics.append(data)
        elif kind == "request":
            self.requests[data["id"]] = data
            while len(self.requests) > MAX_REQUESTS:
                self.requests.pop(next(iter(self.requests)))
            self._requests_dirty = True
        elif kind == "models":
            self._set_models(data)
        elif kind == "log":
            self.query_one("#log", RichLog).write(data)
        elif kind == "status":
            self.server.update(data)
            self._render_status()
        elif kind == "config":
            self.server["env"] = data["env"]
            self._render_status()

    def _set_models(self, models: list[dict]) -> None:
        self.models = models
        self.models_at = time.monotonic()

    # ---- rendering ----

    def refresh_view(self) -> None:
        self._render_status()
        self._render_charts()
        self._render_models()
        if self._requests_dirty:
            self._render_requests()

    def _render_status(self) -> None:
        s = self.server
        conn = "green" if self.connection == "connected" else "red"
        status = s.get("status") or "unknown"
        color = {"running": "green", "starting": "yellow", "restarting": "yellow"}.get(status, "red")
        version = f" v{escape(s['version'])}" if s.get("version") else ""
        machine = escape(" ".join(x for x in (s.get("hostname"), s.get("gpu_name")) if x))
        env = escape("  ".join(f"{k.removeprefix('OLLAMA_')}={v}" for k, v in (s.get("env") or {}).items()))
        self.query_one("#status", Static).update(
            f"[{conn}]●[/] {escape(self.target)} {escape(self.connection)}  │  "
            f"ollama [{color}]{status}[/]{version}  │  {machine}\n[dim]{env}[/]"
        )

    def _render_charts(self) -> None:
        samples = list(self.metrics)
        now = samples[-1]["ts"] if samples else 0.0  # server clock, so clock skew doesn't matter
        xs = [s["ts"] - now for s in samples]
        last = samples[-1] if samples else {}

        def col(key: str, scale: float = 1.0) -> list[float]:
            return [(s.get(key) or 0) / scale for s in samples]

        self.query_one("#cpu", Chart).draw(
            f"CPU {last.get('cpu', 0):.0f}%  ·  ollama {last.get('ollama_cpu', 0):.0f}%",
            [("system", xs, col("cpu")), ("ollama", xs, col("ollama_cpu"))],
            100,
        )
        self.query_one("#ram", Chart).draw(
            f"RAM {_gb(last.get('ram_used'))} / {_gb(last.get('ram_total'))}  ·  ollama {_gb(last.get('ollama_rss'))}",
            [("system", xs, col("ram_used", GB)), ("ollama", xs, col("ollama_rss", GB))],
            (last.get("ram_total") or GB) / GB,
        )
        if last.get("gpu_util") is None:
            gpu_title, vram_title = "GPU n/a", "VRAM n/a"
        else:
            gpu_title = (
                f"GPU {last['gpu_util']}%  ·  {_num(last.get('gpu_temp'))}°C  ·  {_num(last.get('gpu_power'), '{:.0f}')}W"
            )
            vram_title = f"VRAM {_gb(last.get('vram_used'))} / {_gb(last.get('vram_total'))}"
        self.query_one("#gpu", Chart).draw(gpu_title, [("util", xs, col("gpu_util"))], 100)
        self.query_one("#vram", Chart).draw(
            vram_title, [("used", xs, col("vram_used", GB))], (last.get("vram_total") or GB) / GB
        )

    def _render_models(self) -> None:
        table = self.query_one("#models", DataTable)
        table.clear()
        elapsed = time.monotonic() - self.models_at
        for m in self.models:
            pct = m["gpu_pct"]
            table.add_row(
                Text(m["name"] or "?"),
                _gb(m["size"]),
                _gb(m["size_vram"]),
                Text(f"{pct}%", style="green" if pct >= 100 else "yellow"),  # <100% means part runs on CPU
                _num(m.get("context")),
                _countdown(m.get("expires_in"), elapsed),
            )
        if not self.models:
            table.add_row(Text("no model loaded", style="dim"), "", "", "", "", "")

    def _status_cell(self, r: dict) -> Text:
        if r["state"] == "active":
            return Text("▶ active", style="cyan")
        if r["state"] == "cancelled":
            return Text("cancelled", style="yellow")
        return Text(str(r["status"] or "–"), style="red" if r["state"] == "error" else "green")

    def _render_requests(self) -> None:
        self._requests_dirty = False
        table = self.query_one("#requests", DataTable)
        table.clear()
        rows = [r for r in reversed(self.requests.values()) if self.show_all or r["inference"]]
        for r in rows[:SHOWN_REQUESTS]:
            table.add_row(
                time.strftime("%H:%M:%S", time.localtime(r["ts"])),
                Text(r["client"] or ""),
                Text(f"{r['method']} {r['path']}"),
                Text(r["model"] or "–"),
                self._status_cell(r),
                _ms(r["ttft_ms"]),
                _ms(r["duration_ms"]),
                _num(r["prompt_tok"]),
                _num(r["out_tok"]),
                _num(r["tok_s"], "{:.1f}"),
            )

    def _set_requests_title(self) -> None:
        self.query_one("#requests", DataTable).border_title = (
            "Requests (all endpoints)" if self.show_all else "Requests (inference only, f shows all)"
        )

    # ---- actions ----

    async def action_quit(self) -> None:
        if self.owns_server:
            self.push_screen(Confirm("Quit? This also stops Ollama; LLM clients lose access."), self._quit_if_confirmed)
        else:
            self.exit()

    def _quit_if_confirmed(self, ok: bool | None) -> None:
        if ok:
            self.exit()

    def action_toggle_log(self) -> None:
        log = self.query_one("#log", RichLog)
        log.display = not log.display

    def action_toggle_filter(self) -> None:
        self.show_all = not self.show_all
        self._set_requests_title()
        self._render_requests()

    def _post_if_confirmed(self, path: str) -> Callable[[bool | None], None]:
        def callback(ok: bool | None) -> None:
            if ok:
                self._send("POST", path)

        return callback

    def _save_config(self, env: dict[str, str] | None) -> None:
        if env is not None:
            self._send("PUT", "/config", {"env": env})

    def action_toggle_ollama(self) -> None:
        if self.server.get("status") in ("running", "starting", "restarting"):
            self.push_screen(Confirm("Stop Ollama? LLM clients lose access until it is started again."),
                             self._post_if_confirmed("/ollama/stop"))
        else:
            self._send("POST", "/ollama/start")

    def action_restart(self) -> None:
        self.push_screen(Confirm("Restart Ollama?"), self._post_if_confirmed("/ollama/restart"))

    def action_config(self) -> None:
        self.push_screen(ConfigScreen(dict(self.server.get("env") or {})), self._save_config)

    @work(group="control")
    async def _send(self, method: str, path: str, payload: dict | None = None) -> None:
        try:
            status, text = await self.client.request(method, path, payload)
        except Exception as e:  # network errors must not crash the dashboard
            self.notify(f"{method} {path} failed: {e}", severity="error")
            return
        if status >= 400 or status == 0:
            try:
                text = json.loads(text).get("error", text)
            except (ValueError, AttributeError):
                pass
            self.notify(f"{method} {path} failed: {text}", severity="error")
        else:
            self.notify(f"{path.rsplit('/', 1)[-1]}: ok")
