# Crate — setup

Two parts: the **app** (light; runs anywhere) and the optional **analysis pipeline** (heavy, but
runs on any machine — no GPU required). You only need the app to browse, crate, tag, and export.
Add analysis to unlock the map, sonic clusters, harmonic mixing, energy, and colored waveforms.

---

## 1. The app

**Requirements:** Python 3.11+ on Windows (PySide6 also runs on macOS/Linux; `run.ps1` is
Windows — on other OSes create the venv + run `app.py` yourself).

```powershell
# from the crate/ folder
powershell -ExecutionPolicy Bypass -File run.ps1
```

That bootstraps `.venv`, installs `requirements.txt` (PySide6, mutagen, numpy, soundfile), and
opens the window. Then:

1. On first run, point it at your **music folder** when prompted (or set `CRATE_LIB_ROOT`).
2. Click **⚙ FOLDERS** to confirm/add scan roots and your **crates** destination.
3. Click **RE-INDEX** to scan tags into a local `crate.db`.
4. Search, build a crate, **SAVE** it, and export to rekordbox.

`crate.db` and `crate_config.json` are local to your machine (gitignored). Nothing phones home.

### Manual (any OS)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # or .venv/bin/python on macOS/Linux
.venv/Scripts/python app.py
```

---

## 2. Analysis pipeline (optional — for the map, clusters, harmonic mixing, waveforms, energy)

This computes, for each track: BPM/key/energy, a MuQ-MuLan audio embedding, sonic clusters, a 2D+3D
map position (track and artist level), and a colored waveform. It writes SQLite sidecars to
`<lib_root>/.crate/`. The app reads them automatically (RE-INDEX / SYNC pulls BPM/key in; the map
reads the cluster + coord sidecars).

It is **heavy** (pulls torch + the embedding model) but **runs on any machine — no GPU required**:
MuQ-MuLan embeds at ~2s/track on CPU, so a ~1000-track library is a one-time ~30-40 min pass; a GPU
makes it minutes. Use a **separate venv** from the app. (Verified end-to-end on Windows 11 +
Python 3.14, CPU torch.)

```bash
# in a separate venv (run from the crate/ folder)
python -m venv analysis/.venv
analysis/.venv/Scripts/python -m pip install --upgrade pip   # or analysis/.venv/bin/python on macOS/Linux

# 1) install the torch build for your machine FIRST (see https://pytorch.org/get-started/)
#    no GPU / macOS / any laptop:
analysis/.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
#    NVIDIA GPU instead (recent 40/50-series need CUDA 12.8):
#    ... pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) then the rest of the analysis deps
analysis/.venv/Scripts/python -m pip install -r analysis/requirements-analysis.txt

# 3) run the whole pass against your library root
analysis/.venv/Scripts/python analysis/analyze_all.py --root "D:\Music"
```

`analyze_all.py` runs the stages in order: `analyze.py` (BPM/key/energy) → `embed_muq.py` (MuQ-MuLan
vectors) → `cluster.py` (sonic clusters) → `umap_music.py` (track map, 2D+3D) → `umap_artists.py`
(artist map) → `waveform.py` (colored waveforms). Each stage is idempotent — re-running only
processes new/changed files. You can run stages individually with the same `--root`.

**Gotchas:**
- **m4a/AAC** needs `ffmpeg` on `PATH` (libsndfile can't decode it). FLAC/MP3/WAV work without it.
- First run downloads the `OpenMuQ/MuQ-MuLan-large` model from HuggingFace (~hundreds of MB).
- If the app and your music live on different machines, point the app's `CRATE_LIB_ROOT` at the
  same library (e.g. a network mount) so it resolves the `.crate/` sidecars.

---

## Running the app's tests

```powershell
.venv\Scripts\python.exe -m pytest test_library.py -q
```
