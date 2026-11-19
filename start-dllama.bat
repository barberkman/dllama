@echo off
rem Serves Ollama through dllama and shows the dashboard in this window.
rem Quitting the dashboard (q) stops Ollama. If dllama is already serving, this only opens the dashboard.
rem Not named dllama.bat: in this folder, `dllama` would then run this file instead of dllama.exe.

where dllama >nul 2>nul
if errorlevel 1 (
    echo dllama is not installed. In this folder run:
    echo     python -m uv tool install --system-certs .
    echo then open a new terminal.
    pause
    exit /b 1
)

dllama up %*
if errorlevel 1 pause
