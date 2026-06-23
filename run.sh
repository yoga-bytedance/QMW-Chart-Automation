#!/usr/bin/env bash
# Launch QMW Chart Studio.
#   ./run.sh            -> serves on http://127.0.0.1:8000
#   PORT=9000 ./run.sh  -> custom port
set -e
cd "$(dirname "$0")"

# Ensure dependencies (idempotent; installs into the user site only).
if ! python3 - <<'PY' 2>/dev/null
import flask, matplotlib, numpy, scipy, PIL, requests  # noqa
PY
then
  python3 -m pip install --user flask matplotlib numpy scipy pillow requests
fi

export PORT="${PORT:-8000}"
echo "==============================================="
echo "  QMW Chart Studio"
echo "  Open:  http://127.0.0.1:${PORT}"
echo "  Stop:  Ctrl+C"
echo "==============================================="
exec python3 app.py
