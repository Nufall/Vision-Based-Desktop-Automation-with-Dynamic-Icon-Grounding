@echo off
REM Run the post-automation pipeline via uv.
REM First time on a fresh machine: `uv sync` will create .venv and install everything from uv.lock.
pushd "%~dp0"
uv run python main.py %*
popd
