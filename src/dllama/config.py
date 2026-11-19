"""Server and client configuration, stored as TOML under ~/.dllama."""

from __future__ import annotations

import copy
import secrets
import tomllib
from pathlib import Path

import tomli_w

CONFIG_DIR = Path.home() / ".dllama"
SERVER_CONFIG = CONFIG_DIR / "config.toml"
CLIENT_CONFIG = CONFIG_DIR / "client.toml"

SERVER_DEFAULTS: dict = {
    "control": {"bind": "0.0.0.0:11500", "token": ""},
    "proxy": {"bind": "0.0.0.0:11434", "api_key": ""},
    "ollama": {
        "binary": "",
        "internal_port": 11435,
        "autostart": True,
        "env": {
            "OLLAMA_CONTEXT_LENGTH": "65536",
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": "q8_0",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_KEEP_ALIVE": "30m",
        },
    },
}

CLIENT_DEFAULTS: dict = {"host": "169.254.51.51", "port": 11500, "token": ""}


def _merge(defaults: dict, loaded: dict) -> dict:
    """Overlay `loaded` on `defaults`. An `env` table replaces the default one, so keys can be removed."""
    out = copy.deepcopy(defaults)
    for key, value in loaded.items():
        if key != "env" and isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load(path: Path, defaults: dict) -> tuple[dict, bool]:
    if path.exists():
        return _merge(defaults, tomllib.loads(path.read_text(encoding="utf-8"))), False
    return copy.deepcopy(defaults), True


def save(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(cfg), encoding="utf-8")


def load_server(path: Path = SERVER_CONFIG) -> tuple[dict, bool]:
    """Load the server config, creating it with a fresh dashboard token on first run."""
    cfg, created = _load(path, SERVER_DEFAULTS)
    if created or not cfg["control"]["token"]:
        cfg["control"]["token"] = secrets.token_urlsafe(24)
        save(path, cfg)
    return cfg, created


def load_client(path: Path = CLIENT_CONFIG) -> dict:
    """Load the client config, creating it with defaults on first run so it can be edited."""
    cfg, created = _load(path, CLIENT_DEFAULTS)
    if created:
        save(path, cfg)
    return cfg


def split_bind(bind: str) -> tuple[str, int]:
    host, _, port = bind.rpartition(":")
    return host or "0.0.0.0", int(port)


def local_address(bind: str) -> str:
    """Address to reach a server bound to `bind` from the same machine."""
    host = split_bind(bind)[0]
    return "127.0.0.1" if host == "0.0.0.0" else host


def ollama_env(base: dict[str, str], cfg: dict) -> dict[str, str]:
    """Environment for `ollama serve`: inherited OLLAMA_* dropped, config applied, loopback bind forced."""
    env = {k: v for k, v in base.items() if not k.upper().startswith("OLLAMA_")}
    env.update({k: str(v) for k, v in cfg["ollama"]["env"].items()})
    env["OLLAMA_HOST"] = f"127.0.0.1:{cfg['ollama']['internal_port']}"
    return env
