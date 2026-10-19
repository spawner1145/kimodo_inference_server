#!/bin/sh
# Start from this directory even when called from another working directory.
set -eu
SCRIPT_DIR=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

KIMODO_LAUNCH_PYTHON=${KIMODO_PYTHON:-}
if [ -z "$KIMODO_LAUNCH_PYTHON" ] && [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    KIMODO_LAUNCH_PYTHON="$VIRTUAL_ENV/bin/python"
fi
if [ -z "$KIMODO_LAUNCH_PYTHON" ]; then
    for candidate in "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/../.venv-nvfp4/bin/python" "$SCRIPT_DIR/../.venv/bin/python"; do
        if [ -x "$candidate" ]; then
            KIMODO_LAUNCH_PYTHON=$candidate
            break
        fi
    done
fi
if [ -z "$KIMODO_LAUNCH_PYTHON" ]; then
    if command -v python3 >/dev/null 2>&1; then
        KIMODO_LAUNCH_PYTHON=python3
    else
        KIMODO_LAUNCH_PYTHON=python
    fi
fi

export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
printf 'Python: %s\n' "$KIMODO_LAUNCH_PYTHON"
if ! "$KIMODO_LAUNCH_PYTHON" -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ is required'; import fastapi, uvicorn, torch, kimodo, kimodo_server"; then
    printf '\n[ERROR] Python or the Kimodo server dependencies are unavailable.\n' >&2
    printf 'Set KIMODO_PYTHON to a global or virtual environment Python with the dependencies installed.\nSee README.md for setup. Nothing was installed or downloaded.\n' >&2
    exit 1
fi

printf 'Starting Kimodo. Default WebUI: http://127.0.0.1:6600/\n'
printf 'The listening address follows config.json and command-line overrides.\n'
printf 'Press Ctrl+C to stop. No models will be downloaded automatically.\n'
exec "$KIMODO_LAUNCH_PYTHON" -m kimodo_server serve --config "$SCRIPT_DIR/config.json" "$@"
