# dllama

Runs Ollama with a fixed configuration on one machine and shows a live terminal dashboard on any other machine on the network.

```
 LLM clients ──► :11434  dllama proxy ──► 127.0.0.1:11435  ollama serve (started by dllama)
 dllama connect ──► :11500  dashboard API (token protected)
```

- **`dllama serve`** (on the Ollama machine) starts `ollama serve` with your settings, restarts it if it crashes, proxies the Ollama API so every request is logged, and samples CPU, RAM and NVIDIA GPU usage.
- **`dllama connect`** (on any machine) opens the dashboard: live CPU/RAM/GPU/VRAM charts, loaded models (with the GPU/CPU split), every request with tokens and tok/s, and the Ollama log. Closing it does not affect Ollama.

## Server setup (the Ollama machine)

1. Install Ollama, then **quit the tray app** (tray icon > Quit Ollama) and turn off its start-at-login (remove the Ollama shortcut from `shell:startup`). Otherwise it takes port 11434. `setx OLLAMA_*` variables are not needed; dllama ignores them.
2. Install [uv](https://docs.astral.sh/uv/), copy this folder over, and in it run:
   ```powershell
   uv tool install .
   ```
   (Behind a TLS-inspecting proxy, add `--system-certs`.) If `dllama` is not found afterwards, run `uv tool update-shell` and open a new terminal.
3. Allow the two ports through the firewall (run as administrator). `LocalSubnet` works for link-local addresses such as `169.254.x.x`, which Windows usually puts on the Public profile:
   ```powershell
   New-NetFirewallRule -DisplayName dllama -Direction Inbound -Protocol TCP -LocalPort 11434,11500 -RemoteAddress LocalSubnet -Action Allow
   ```
4. Run it:
   ```powershell
   dllama serve
   ```
   The first run creates `~\.dllama\config.toml` and prints the machine's addresses and the **dashboard token**.

   To watch it on the same machine, run `dllama up` instead (or double-click `start-dllama.bat`): it serves and shows the dashboard in the same terminal. Quitting that dashboard stops Ollama. If `dllama serve` is already running, `dllama up` only opens the dashboard.

To start it at logon (in your user session, so it can see the GPU):

```powershell
$action   = New-ScheduledTaskAction -Execute (Get-Command dllama).Source -Argument serve
$trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName dllama -Action $action -Trigger $trigger -Settings $settings
```

### Server config (`~\.dllama\config.toml`)

```toml
[control]
bind  = "0.0.0.0:11500"     # dashboard API
token = "..."               # generated on first run

[proxy]
bind    = "0.0.0.0:11434"   # what LLM clients connect to (replaces OLLAMA_HOST)
api_key = ""                # optional: require "Authorization: Bearer <key>" or "x-api-key: <key>"

[ollama]
binary        = ""          # empty: PATH, then %LOCALAPPDATA%\Programs\Ollama\ollama.exe
internal_port = 11435       # ollama listens on 127.0.0.1 only
autostart     = true

[ollama.env]                # passed to `ollama serve`; editable from the dashboard (c)
OLLAMA_CONTEXT_LENGTH  = "65536"
OLLAMA_FLASH_ATTENTION = "1"
OLLAMA_KV_CACHE_TYPE   = "q8_0"
OLLAMA_NUM_PARALLEL    = "1"
OLLAMA_KEEP_ALIVE      = "30m"
```

Restart `dllama serve` after editing `[control]`, `[proxy]` or `[ollama]` by hand. `[ollama.env]` changes made from the dashboard apply immediately (Ollama restarts).

## Client setup (the computer you watch from)

Install the same way (`uv tool install .`), then:

```powershell
dllama connect
```

The first run creates `~\.dllama\client.toml`:

```toml
host  = "169.254.51.51"   # the Ollama machine
port  = 11500
token = ""                # paste the token printed by `dllama serve`
```

`dllama connect <host> --port <port> --token <token>` overrides the file, and `DLLAMA_TOKEN` overrides the file's token.

Point your LLM tools at `http://169.254.51.51:11434` (the native Ollama API, plus the OpenAI- and Anthropic-compatible endpoints).

### Keys

| Key | Action |
|---|---|
| `s` | start / stop Ollama (stop asks first) |
| `r` | restart Ollama (asks first) |
| `c` | edit the Ollama environment, save restarts Ollama |
| `l` | show / hide the Ollama log |
| `f` | requests: inference only / all endpoints |
| `q` | quit the dashboard (Ollama keeps running; under `dllama up` it stops, so it asks first) |

"On GPU" below 100% (yellow) means part of the model runs on the CPU. Lower the context length or use a smaller model.

## Notes

- Traffic is plain HTTP. The dashboard token and API key travel unencrypted, which is fine on a trusted LAN.
- Everything (metric history, last 500 requests, last 1000 log lines) lives in memory and is gone when `dllama serve` stops.
- Killing `dllama serve` (Ctrl+C, Task Manager, crash) also kills Ollama and its runners, freeing VRAM.

## Development

```powershell
uv sync
uv run pytest
```
