# crate.spec — self-contained two-exe build:
#   Crate.exe          the PySide6 GUI (light: Qt + numpy + mutagen)
#   crate-analyze.exe  the heavy local analysis pipeline (torch + MuQ + librosa + umap …)
# Both land in dist/Crate/. The GUI shells crate-analyze.exe (sibling) for ANALYZE; the MuQ model
# auto-downloads to the user's HF cache on first analyze (not bundled — it's ~3.8 GB).
#
# Build:  analysis/.venv/Scripts/pyinstaller crate.spec --noconfirm
import os
from PyInstaller.utils.hooks import collect_all

HERE = os.path.abspath(os.getcwd())
ANALYSIS = os.path.join(HERE, "analysis")

# ---- data shipped with the app (read-only; resolved via _MEIPASS at runtime) ----
shared_datas = [("skins", "skins"), ("assets", "assets")]

# ---- GUI exe: trace app.py; keep the heavy ML libs OUT (app never imports them) ----
gui = Analysis(
    ["app.py"],
    pathex=[HERE],
    datas=shared_datas,
    hiddenimports=["map_view", "library", "theme", "tag_drawer", "waveform_view"],
    excludes=["torch", "torchaudio", "torchvision", "transformers", "muq", "librosa",
              "numba", "llvmlite", "sklearn", "scikit_learn", "umap", "hdbscan", "pacmap",
              "nnAudio", "x_clip", "faiss", "tensorflow", "matplotlib"],
    noarchive=False,
)

# ---- analysis exe: bundle the whole heavy stack ----
an_datas, an_binaries, an_hidden = list(shared_datas), [], [
    # step modules are loaded by path / importlib, so PyInstaller can't see them statically
    "analyze_all", "_common", "analyze", "embed_muq", "cluster",
    "umap_music", "umap_artists", "waveform", "vectors",
]
for pkg in ["torch", "torchaudio", "transformers", "tokenizers", "safetensors",
            "muq", "nnAudio", "x_clip", "einops", "beartype", "easydict",
            "librosa", "soundfile", "soxr", "audioread", "lazy_loader", "pooch",
            "sklearn", "scipy", "numba", "llvmlite", "umap", "pynndescent",
            "hdbscan", "pacmap", "faiss", "huggingface_hub", "pyloudnorm",
            "joblib", "threadpoolctl", "decorator", "msgpack", "cffi", "regex"]:
    try:
        d, b, h = collect_all(pkg)
        an_datas += d; an_binaries += b; an_hidden += h
    except Exception as e:
        print(f"crate.spec: collect_all({pkg}) skipped: {e}")

analyze = Analysis(
    [os.path.join("analysis", "crate_analyze_entry.py")],
    pathex=[HERE, ANALYSIS],
    binaries=an_binaries,
    datas=an_datas,
    hiddenimports=an_hidden,
    excludes=["PySide6", "shiboken6", "tkinter", "matplotlib"],
    noarchive=False,
)

gui_pyz = PYZ(gui.pure)
an_pyz = PYZ(analyze.pure)

gui_exe = EXE(
    gui_pyz, gui.scripts, [], exclude_binaries=True,
    name="Crate", console=False,
    icon=os.path.join("assets", "crate_icon.ico"),
)
an_exe = EXE(
    an_pyz, analyze.scripts, [], exclude_binaries=True,
    name="crate-analyze", console=True,
)

coll = COLLECT(
    gui_exe, gui.binaries, gui.datas,
    an_exe, analyze.binaries, analyze.datas,
    strip=False, upx=False, name="Crate",
)
