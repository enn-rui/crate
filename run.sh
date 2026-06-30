#!/usr/bin/env bash
# run.sh — launch Crate from source on macOS / Linux.
#   chmod +x run.sh && ./run.sh
# Bootstraps a local .venv on first run (the light GUI deps only — the heavy analysis stack is a
# separate, optional install; see analysis/requirements-analysis.txt), then opens the app window.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PY="$VENV/bin/python"
OK="$VENV/.deps-ok"   # written only after a clean install (so a half install retries, not launches)

if [[ ! -f "$OK" ]]; then
  echo "First run: creating .venv and installing dependencies..."
  PYBIN="$(command -v python3 || command -v python || true)"
  if [[ -z "$PYBIN" ]]; then
    echo "ERROR: Python 3.11+ not found. Install it (https://www.python.org/downloads/) and re-run." >&2
    exit 1
  fi
  VER="$("$PYBIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if ! "$PYBIN" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,11) else 1)'; then
    echo "ERROR: Crate needs Python 3.11+, found $VER. Install a newer Python and re-run." >&2
    exit 1
  fi
  [[ -x "$PY" ]] || "$PYBIN" -m venv "$VENV"
  "$PY" -m pip install --upgrade pip
  if ! "$PY" -m pip install -r "$HERE/requirements.txt"; then
    echo "ERROR: dependency install failed (see pip output above). Fix it (usually network) and re-run." >&2
    exit 1
  fi
  touch "$OK"
  echo "Setup complete."
fi

exec "$PY" "$HERE/app.py"
