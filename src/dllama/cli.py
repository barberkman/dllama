"""Command line entry point: `dllama serve`, `dllama up` and `dllama connect`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import config


def main() -> None:
    parser = argparse.ArgumentParser(prog="dllama", description="Ollama supervisor with a remote terminal dashboard.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run ollama, the API proxy and the dashboard server (on the Ollama machine)")
    serve.add_argument("--config", type=Path, default=config.SERVER_CONFIG, help="server config (default: %(default)s)")

    up = sub.add_parser("up", help="serve and show the dashboard in this terminal; quitting it stops ollama")
    up.add_argument("--config", type=Path, default=config.SERVER_CONFIG, help="server config (default: %(default)s)")

    connect = sub.add_parser("connect", help="open the dashboard of a dllama server")
    connect.add_argument("host", nargs="?", help="server address (default: `host` in the client config)")
    connect.add_argument("--port", type=int, help="dashboard port (default: `port` in the client config)")
    connect.add_argument("--token", help="dashboard token (default: $DLLAMA_TOKEN, then `token` in the client config)")
    connect.add_argument("--config", type=Path, default=config.CLIENT_CONFIG, help="client config (default: %(default)s)")

    args = parser.parse_args()
    if args.command == "serve":
        from .server import run

        run(args.config)
        return
    if args.command == "up":
        from .server import run_with_dashboard

        run_with_dashboard(args.config)
        return

    cfg = config.load_client(args.config)
    token = args.token or os.environ.get("DLLAMA_TOKEN") or cfg["token"]
    if not token:
        sys.exit(
            f"No dashboard token. Put the token printed by `dllama serve` into {args.config} "
            "(token = \"...\"), set DLLAMA_TOKEN, or pass --token."
        )
    from .tui import Dashboard

    Dashboard(args.host or cfg["host"], args.port or cfg["port"], token).run()
