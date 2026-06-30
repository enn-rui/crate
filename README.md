# Crate — DJ set prep + discovery

A native PySide6 desktop app for prepping DJ sets: browse your lossless library, build crates,
tag/cue tracks, explore your collection on a **MuQ-MuLan audio-embedding map** (a sonic-cluster
view of your library that no other DJ tool ships), and export a **rekordbox**-ready playlist
(XML + `.m3u8`) with local copies.

Built for a DDJ-FLX4 + rekordbox workflow, but the library/crate/export side works with any
local music folder and any DJ app that imports `.m3u8` or rekordbox XML.

## Quick start

**Option A — download the standalone app (Windows, no Python needed).** Grab the latest
`Crate-win.zip` from Releases, unzip, and double-click `Crate.exe`. Everything is bundled,
including the analysis pipeline; the embedding model (~3.8 GB) downloads itself the first time you
hit **ANALYZE** and is cached forever after. Everything else works fully offline.

**Option B — run from source** (any OS with Python 3.11+):

```bash
./run.ps1        # Windows  (powershell -ExecutionPolicy Bypass -File run.ps1)
./run.sh         # macOS / Linux  (chmod +x run.sh first)
```

First run creates a local `.venv`, installs `requirements.txt`, and opens the window. On a fresh
machine it asks you to point at your music folder, then **RE-INDEX** to scan it into a local
`crate.db`. (See `SETUP.md` for the full setup, including the audio-analysis pipeline.)

## What it does

- **LIST** — searchable/sortable table (artist/title/album, BPM, key, rating, color, tags, cue,
  comment). Filter by bucket, BPM range, and Camelot key. `◇ MIXES WITH` shows harmonically
  compatible tracks for the selected one.
- **MAP** — a 2D/3D map of your library from MuQ-MuLan audio embeddings (PaCMAP projection),
  with sonic clusters found in the full 512-d space (HDBSCAN). Dots = tracks, color by
  cluster / key / tempo, sized by rating. `CONNECT` (sonic / key / tempo / artist) re-lays-out
  around a relationship. Click a dot to play it; the map walks nearest-neighbour for discovery.
- **ARTISTS** — toggle the map to an artist-level scatter (one dot per artist, sized by track
  count). Click an artist to filter the list down to their tracks.
- **Tags & cues** — rekordbox-style My-Tags (genre / components / vocal / situation / mood), free
  tags, color label, comment, and memory/hot cues placed on a colored waveform. Auto-saved.
- **Crates** — assemble a crate, **SAVE** it as a reopenable folder (local copies + `.m3u8` +
  rekordbox `.xml`). Reopen any saved crate from the dropdown.
- **rekordbox export** — the saved `.xml` carries your prep (rating, color, tags→comments, memory
  & hot cues). BPM/key are intentionally omitted so rekordbox builds its own beatgrid on import.
  In rekordbox: enable *Preferences ▸ View ▸ rekordbox xml*, then import the playlist.
- **Skins** — pick a skin in the top bar (`terminal`, `winamp`, `transflag`). Add your own by
  dropping a `skins/<name>.qss` in.

## Configuration

The library root and crate destination are set in-app (**⚙ FOLDERS** / first-run prompt) and saved
to a local `crate_config.json`. You can also override via env vars:

- `CRATE_LIB_ROOT` — library root the app scans and resolves analysis sidecars under.
- `CRATE_EXPORT_ROOT` — default export folder (default `%USERPROFILE%\Music\DJ\incoming`).

## Audio analysis (optional, for BPM/key/energy + the maps)

The search/crate/export features work from file tags alone. The **map**, harmonic mixing, colored
waveforms, and energy need an analysis pass (BPM/key/energy + MuQ-MuLan embeddings + clusters +
map). It lives in `analysis/` and writes sidecar SQLite files next to your music
(`<lib_root>/.crate/`). The app reads those sidecars; it does not require them to run.

It's heavy (pulls torch + the embedding model) but **runs on any machine, no GPU required** —
MuQ-MuLan embeds at ~2s/track on CPU (a ~1000-track library is a one-time ~30-40 min pass); a GPU
just makes it minutes. In the **standalone app** the whole pipeline is already bundled — just press
**ANALYZE** (the model auto-downloads once). From source, set up the analysis venv per `SETUP.md`,
then press ANALYZE or run `analysis/analyze_all.py --root <your music>` directly.

## Build a standalone app

To produce your own self-contained build (bundles the GUI + the full analysis pipeline into one
folder; the model still downloads on first analyze):

```bash
python -m venv analysis/.venv
analysis/.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
analysis/.venv/Scripts/pip install -r analysis/requirements-analysis.txt
analysis/.venv/Scripts/pip install "PySide6>=6.11" mutagen pyinstaller
analysis/.venv/Scripts/pyinstaller crate.spec --noconfirm --clean
# -> dist/Crate/  (Crate.exe + crate-analyze.exe + everything they need)
```

The recipe is `crate.spec`. On macOS/Linux use the `bin/` venv paths and a CUDA torch wheel if you
have an NVIDIA GPU. The build is ~1.2 GB; zip `dist/Crate/` to share.

## Layout

- `library.py` — pure logic: index / search / export / crates / tags / cues / rekordbox XML
  (no Qt; unit-tested in `test_library.py`).
- `app.py` — the PySide6 window (LIST / MAP modes, transport, crate panel).
- `map_view.py` — the track map (`MapView`) and artist map (`ArtistMapView`).
- `tag_drawer.py` / `waveform_view.py` — the tag inspector and waveform/cue widgets.
- `theme.py` + `skins/*.qss` — skin system.
- `analysis/` — the offline analysis pipeline (run separately; see `SETUP.md`).
- `crate.db`, `crate_config.json`, `.venv/` — local, per-machine; gitignored.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest test_library.py -q
```
