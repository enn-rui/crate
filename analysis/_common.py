"""Shared config for Crate's local analysis scripts.

These are the box's analysis programs, vendored to run on ONE computer pointing at a local music
library root. They resolve a single library ROOT and write all sidecars
into ROOT/.crate/ keyed by path RELATIVE TO ROOT (forward slashes) — exactly how the Crate app
(library.py) reads them back, so analysis on the friend's PC drops straight into the app.

ROOT resolution order: --root arg  >  CRATE_LIB_ROOT env  >  lib_root in ../crate_config.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

AUDIO_EXTS = {".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wav", ".aiff", ".aif"}


def resolve_root(arg_root: str | None = None) -> Path:
    if arg_root:
        return Path(arg_root)
    env = os.environ.get("CRATE_LIB_ROOT")
    if env:
        return Path(env)
    cfg = Path(__file__).resolve().parent.parent / "crate_config.json"
    if cfg.exists():
        try:
            v = json.loads(cfg.read_text(encoding="utf-8")).get("lib_root")
            if v:
                return Path(v)
        except Exception:
            pass
    raise SystemExit("No library root. Pass --root <music folder>, set CRATE_LIB_ROOT, "
                     "or run Crate once and set your music folder in ⚙ FOLDERS.")


def crate_dir(root: Path) -> Path:
    d = Path(root) / ".crate"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_buckets(arg: str | None) -> set[str] | None:
    """'music,dj' -> {'music','dj'}; None/'' -> None (no filter)."""
    if not arg:
        return None
    return {b.strip() for b in arg.split(",") if b.strip()}


def iter_audio(root: Path, buckets: set[str] | None = None):
    """Yield every audio file under ROOT (recursive), skipping the .crate sidecar dir.

    If `buckets` is given, only files whose FIRST path segment under ROOT is in the set are
    yielded — so analysis can target the real library folders (e.g. music/dj/music-mp3) and skip
    download caches / quarantine dirs (slskd/, _youtube_quarantine/, downloads/) sharing the root.
    """
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in AUDIO_EXTS and ".crate" not in p.parts):
            continue
        if buckets is not None:
            rel = p.relative_to(root).parts
            if not rel or rel[0] not in buckets:
                continue
        yield p
