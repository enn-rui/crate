# Crate — agent setup runbook

**Audience:** the AI coding agent helping a new user set Crate up on a fresh machine.
**Goal:** get the app running, then (optionally) the heavy analysis pipeline. Follow this top to bottom.

Crate is a native **PySide6** desktop app for DJ set prep + library discovery. It indexes a folder of
local audio into a per-machine SQLite DB and exports rekordbox-ready crates. See `README.md` for the
feature pitch; this file is just setup. **Nothing phones home; all data stays local.**

---

## 0. Mental model (read first — it determines what you install)

Crate has **two tiers**, and the user only needs the first one to get value:

| Tier | What it gives | Cost | Required? |
|------|---------------|------|-----------|
| **App** | Browse/search/tag, build crates, ratings/cues, rekordbox XML + M3U8 export | light (PySide6, ~5 deps) | **Yes** |
| **Analysis** | The MuQ-MuLan "sonic map" + clusters, harmonic/COMPATIBLE mixing, energy, colored waveforms | heavy (torch + a model download), but runs on CPU — no GPU required | **No — optional** |

The **app reads** analysis results as SQLite "sidecars" under `<library_root>/.crate/`. The **analysis
pipeline writes** them. The two tiers are decoupled, which means two valid topologies:

- **Topology A — one machine (simplest).** Run both tiers on the same PC. The library folder is local;
  analysis writes `.crate/` right next to the music. **Start here unless the user says otherwise.**
- **Topology B — split (optional).** The app runs on a workstation; the music library + analysis live
  on a separate GPU box, exposed to the workstation as a mounted drive/share. Point the app's library
  root at that mount so it resolves the same `.crate/` sidecars. You only need this if you actually have
  a separate analysis machine — **do not assume it.** (Any specific paths in docs/examples are just
  examples; the pipeline is fully portable via `--root`.)

If the user just wants to try it: do Topology A, **app tier only**, and skip analysis. The map/mix
features will be inert until analysis runs, and the app handles that empty state gracefully.

---

## 1. App tier (do this first)

### Prerequisites
- **Python 3.11+ on PATH** (`python --version`). On Windows, install from python.org with
  "Add python.exe to PATH" ticked. The launcher checks the version and errors clearly if it's missing.
- Windows is the first-class target (`run.ps1`). macOS/Linux work too — use the manual steps below.

### Windows (one command)
```powershell
# from the crate/ folder
powershell -ExecutionPolicy Bypass -File run.ps1
```
`run.ps1` creates `.venv`, installs `requirements.txt` (PySide6, mutagen, numpy, soundfile), writes a
`.venv\.deps-ok` sentinel on success, and launches the window. If the install fails partway it prints
the error and retries on the next run (it will **not** silently launch a broken app).

### macOS / Linux (manual)
```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

### Verify the app tier
```powershell
.venv\Scripts\python.exe -m pytest test_library.py -q      # expect: 53 passed
```
Then the window should open. **First-run flow inside the app:**
1. Point it at the user's **music folder** when prompted (or set the `CRATE_LIB_ROOT` env var).
2. **⚙ FOLDERS** → confirm scan roots + a **crates** output folder → **SAVE & RE-INDEX**.
3. Search, build a crate, **SAVE**, and **EXPORT** (writes `.m3u8` + an `.xml` rekordbox can import).

At this point everything except the map/mix/waveform features works. Those need tier 2.

---

## 2. Analysis tier (optional — the map, COMPATIBLE mixing, waveforms, energy)

Only do this if the user wants the sonic map / harmonic mixing. It is **heavy** (pulls torch +
the embedding model, downloads `OpenMuQ/MuQ-MuLan-large` on first run, ~hundreds of MB) but **runs
on any machine — no GPU required** (MuQ-MuLan embeds at ~2s/track on CPU; a ~1000-track library is a
one-time ~30-40 min pass, minutes on a GPU). Use a **separate venv** from the app. (Proven
end-to-end on Windows 11 + Python 3.14, CPU torch.)

```bash
# in a SEPARATE venv (run from the crate/ folder)
python -m venv analysis/.venv
analysis/.venv/Scripts/python -m pip install --upgrade pip

# 1) install the torch build for the machine FIRST — see https://pytorch.org/get-started/
#    no GPU / macOS / any laptop (the safe default):
analysis/.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
#    NVIDIA GPU instead (recent 40/50-series need CUDA 12.8):
#    ... pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) then the rest of the analysis deps
analysis/.venv/Scripts/python -m pip install -r analysis/requirements-analysis.txt

# 3) run the whole pass against the library root
analysis/.venv/Scripts/python analysis/analyze_all.py --root "D:\Music"
```

`analyze_all.py` runs the stages in order and each is **idempotent** (re-running only processes
new/changed files): `analyze.py` (BPM/key/energy) → `embed_muq.py` (MuQ-MuLan vectors) →
`cluster.py` (sonic clusters) → `umap_music.py` (track map, 2D+3D) → `umap_artists.py` (artist map)
→ `waveform.py` (colored waveforms). It writes the sidecars to `<root>/.crate/`. Then in the app hit
**SYNC** (pulls BPM/key) and switch to **MAP**.

**Wiring the app's in-app ANALYZE button (optional):** set `"analysis_python"` in `crate_config.json`
to the `analysis/.venv` interpreter path; the **ANALYZE** button then shells the pipeline for you. If
that key is absent the button just shows a "here's how to set it up" dialog.

**Or run analysis on a separate box over SSH (no local analysis venv):** set `"analysis_remote"` —
the box does all the compute and this machine just triggers it, then reads the sidecars over the
mounted library. Needs key-based SSH (no password prompt) + `ssh` on PATH. Takes precedence over
`analysis_python` when set.

```json
"analysis_remote": { "ssh": "user@host", "python": "~/crate/.venv/bin/python",
                     "script": "~/crate/analysis/analyze_all.py", "root": "/path/to/music" }
```

Only `ssh` is required; the rest default to the layout above.

### Gotchas
- **m4a/AAC** decoding needs **ffmpeg on PATH** (libsndfile can't decode AAC). FLAC/WAV/MP3 are fine
  without it.
- **First run** downloads the `OpenMuQ/MuQ-MuLan-large` model — expect a one-time delay.
- **Tiny test libraries are fine.** If you verify the pipeline on just a few tracks, the UMAP stages
  now exit successfully with placeholder coords (they need ≥5 tracks / ≥3 artists for a real map) —
  a small library will **not** fail the pass.
- **Split topology:** if the app and the music are on different machines, point the app's
  `CRATE_LIB_ROOT` (or in-app library root) at the same mounted library so it finds the `.crate/`
  sidecars the analysis machine wrote.

### macOS / Apple Silicon (M-series) — read this if you're on a Mac

The app tier is plain PySide6 and **just works** on macOS: `chmod +x run.sh && ./run.sh`. The only part
that needs care is the **analysis tier**, because the MuQ stack (torch / torchaudio / nnAudio / numba) is
newer than what's been verified on Apple Silicon. It almost certainly works — these are the knobs if it
fights you:

1. **Use Python 3.12 or 3.13, not 3.14, for the analysis venv.** `librosa` needs `numba`, and
   `numba`/`llvmlite` wheels lag the newest Python — on macOS ARM, 3.12/3.13 have prebuilt wheels;
   3.14 may try to build from source and fail. (The app tier is fine on any 3.11+.) If you only have
   3.14, `brew install python@3.13` and make the analysis venv with that one.
2. **Install torch the macOS way — NO `--index-url`.** On Apple Silicon the default PyPI wheels are the
   right (arm64, MPS-capable) ones. The `download.pytorch.org/whl/cpu` index in the Windows steps above
   is **Windows/Linux only** — don't use it on a Mac.
   ```bash
   python3.13 -m venv analysis/.venv
   analysis/.venv/bin/pip install --upgrade pip
   analysis/.venv/bin/pip install torch torchaudio          # default arm64 wheels
   analysis/.venv/bin/pip install -r analysis/requirements-analysis.txt
   brew install ffmpeg                                       # for m4a/AAC decoding
   analysis/.venv/bin/python analysis/analyze_all.py --root "$HOME/Music/<your library>"
   ```
3. **MuQ runs on CPU here — that's fine.** It auto-uses CUDA only on NVIDIA; on a Mac it embeds on CPU
   at ~2 s/track (a ~1000-track library is a one-time ~30–40 min pass). The model downloads from
   Hugging Face on the first run (needs network once, then cached).
4. **If a MuQ dep simply won't install, you lose nothing essential.** The whole app — browse, crates,
   tags/cues, rekordbox export — works without the analysis tier; only the map / colored waveforms /
   COMPATIBLE mixing stay empty. Ship the app, get analysis working separately. (If `nnAudio` or
   `x_clip` choke, retry after the Python-version fix in (1) — that resolves the usual culprit.)

**A standalone `.app` (optional):** the same `crate.spec` builds on a Mac
(`analysis/.venv/bin/pip install "PySide6>=6.11" mutagen pyinstaller` then
`analysis/.venv/bin/pyinstaller crate.spec --noconfirm --clean`) and produces a `dist/Crate/` folder with
runnable `Crate` + `crate-analyze` binaries. For a double-clickable `.app` bundle you'd add a `BUNDLE()`
block to the spec (not yet set up) — but **for a Mac, just running from source via `run.sh` is the easy,
recommended path.** No build needed to use it.

---

## 3. What you do NOT need

Crate needs nothing beyond your own music folder (plus the analysis venv if you want the map/waveforms).
If you see any references to a separate "box", a network drive, or a media server in older notes, those
describe an optional split topology — not a requirement. A single machine with your music is all it takes.

---

## 4. Quick troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `run.ps1` errors "Python not found" | Install Python 3.11+ and tick "Add to PATH", then re-run. |
| App launches then nothing / blank | A prior half-install. Delete `.venv\`, re-run `run.ps1` (clean retry). |
| MAP is empty | Analysis hasn't run for this library. Run tier 2, then SYNC. Expected without tier 2. |
| Map/mix features inert, no errors | Same as above — app tier alone has no embeddings. Not a bug. |
| m4a tracks fail in analysis | Install `ffmpeg` and put it on PATH. |
| "Folders unreachable" on RE-INDEX | A scan root (e.g. a network drive) isn't mounted. Reconnect it; the index/ratings are kept safe. |
