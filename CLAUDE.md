# Crate — agent guide

Native **PySide6** desktop app for DJ set prep (see `README.md` for features, `SETUP.md` to set up).

## Architecture
- **`library.py`** — all logic (index/search/export, crates, tags, cues, ratings, harmonic mixing,
  rekordbox XML). **No Qt imports.** Unit-tested in `test_library.py` — keep it that way.
- **`app.py`** — the Qt window (LIST/MAP modes, transport, crate panel). UI only; it calls into
  `library`. No business logic here.
- **`map_view.py`** — `MapView` (track UMAP scatter) + `ArtistMapView` (artist scatter).
- **`tag_drawer.py`**, **`waveform_view.py`** — tag inspector + waveform/cue widgets.
- **`theme.py`** + **`skins/*.qss`** — skin system (drop a new `.qss` in to add a skin).
- **`analysis/`** — the offline BPM/key/energy + MuQ-MuLan embedding + clusters + map + waveform
  pipeline. Runs in its own
  venv (`requirements-analysis.txt`), writes SQLite sidecars to `<lib_root>/.crate/`. The app reads
  those; it does not run analysis itself.

## Conventions
- Keep logic in `library.py` (testable, Qt-free); the GUI only calls into it.
- Long operations (file copy, indexing) run off the UI thread.
- Local/per-machine files are gitignored: `crate.db`, `crate_config.json`, `.venv/`, sidecars.
- Run tests after changing `library.py`: `.venv\Scripts\python.exe -m pytest test_library.py -q`.
- The library root is configurable (in-app ⚙ FOLDERS / first-run / `CRATE_LIB_ROOT`); don't hardcode
  paths. Analysis sidecars resolve under `<lib_root>/.crate/`.
