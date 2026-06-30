"""Theme loader — applies a *skin* (a .qss file under skins/) to the Qt app.

Each skin is a single self-documenting stylesheet in skins/<key>.qss with a header comment
that declares its display name, preferred fonts, and whether it wants the scanline texture:

    /* @name     Winamp Classic
       @font     Tahoma, Arial, sans-serif
       @scanline off */

At load the loader substitutes three tokens in the stylesheet text:
    __FONT__      -> the first @font family actually installed (else a sane fallback)
    __SCANLINE__  -> path to the generated scanline tile  (only meaningful if @scanline on)
    __ASSETS__    -> absolute posix path to the skins/ folder, for url(...) image refs

To add a skin, drop a new skins/<name>.qss in — it shows up in the app's skin picker
automatically. Fails safe: if anything here breaks, the app still runs (just unstyled).
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

from PySide6.QtGui import QImage, QColor, QFont, QFontDatabase

# skins/ ship WITH the app (read-only when frozen -> _MEIPASS); the generated scanline tile is
# written to a WRITABLE cache dir so it works inside a frozen / read-only install.
if getattr(sys, "frozen", False):
    HERE = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    _CACHE = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "Crate"
else:
    HERE = Path(__file__).parent
    _CACHE = HERE / "assets"
SKINS_DIR = HERE / "skins"
SCANLINE_PATH = _CACHE / "scanline.png"
DEFAULT_SKIN = "terminal"

# fallback monospace stack if a skin's @font families are all missing
MONO_CANDIDATES = ["IBM Plex Mono", "Cascadia Mono", "Cascadia Code",
                   "Consolas", "DejaVu Sans Mono", "Courier New"]


def _parse_meta(text: str) -> dict:
    """Pull @name / @font / @scanline out of the leading comment header."""
    meta = {"name": None, "fonts": [], "scanline": False}
    head = text[:1200]
    if m := re.search(r"@name\s+(.+)", head):
        meta["name"] = m.group(1).strip().strip("*").strip()
    if m := re.search(r"@font\s+(.+)", head):
        meta["fonts"] = [f.strip() for f in m.group(1).strip().strip("*").split(",") if f.strip()]
    if m := re.search(r"@scanline\s+(on|off|true|false)", head, re.I):
        meta["scanline"] = m.group(1).lower() in ("on", "true")
    return meta


def list_skins() -> list[tuple[str, str]]:
    """[(key, display_name)] for every skins/*.qss, the default first then alphabetical."""
    out = []
    if SKINS_DIR.exists():
        for p in sorted(SKINS_DIR.glob("*.qss")):
            try:
                name = _parse_meta(p.read_text(encoding="utf-8")).get("name") or p.stem.title()
            except Exception:
                name = p.stem.title()
            out.append((p.stem, name))
    out.sort(key=lambda kv: (kv[0] != DEFAULT_SKIN, kv[1].lower()))
    return out


def pick_font(candidates: list[str]) -> str:
    fams = set(QFontDatabase.families())
    for f in list(candidates) + MONO_CANDIDATES:
        if f and f in fams:
            return f
    return "Segoe UI"


def ensure_scanline(path: Path) -> Path:
    """Write a 2x3 ARGB tile: one faint horizontal line every 3px. Tiled by QSS."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    img = QImage(2, 3, QImage.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    for x in range(2):
        img.setPixelColor(x, 0, QColor(255, 255, 255, 10))
    img.save(str(path), "PNG")
    return path


def apply(app, skin_key: str | None = None) -> str:
    """Apply the named skin (or the default). Returns the key actually applied."""
    key = skin_key or DEFAULT_SKIN
    qss_path = SKINS_DIR / f"{key}.qss"
    if not qss_path.exists():
        qss_path = SKINS_DIR / f"{DEFAULT_SKIN}.qss"
        key = DEFAULT_SKIN
    try:
        text = qss_path.read_text(encoding="utf-8")
        meta = _parse_meta(text)
        family = pick_font(meta["fonts"])
        f = QFont(family)
        if any("mono" in c.lower() or "consolas" in c.lower() for c in meta["fonts"]):
            f.setStyleHint(QFont.Monospace)
        app.setFont(f)

        scan = str(ensure_scanline(SCANLINE_PATH)).replace("\\", "/") \
            if meta["scanline"] else ""
        text = (text.replace("__FONT__", family)
                    .replace("__SCANLINE__", scan)
                    .replace("__ASSETS__", str(SKINS_DIR).replace("\\", "/")))
        app.setStyleSheet(text)
    except Exception as e:  # never let theming crash the app
        print(f"theme: skipped ({e})")
    return key
