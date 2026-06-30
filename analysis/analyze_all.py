#!/usr/bin/env python3
"""analyze_all.py — run Crate's full local analysis pipeline, in order.

  1. analyze.py      BPM / musical key (Camelot) / energy           -> features.sqlite
  2. embed_muq.py    MuQ-MuLan music audio vectors (GPU if present) -> music_vectors.sqlite
  3. cluster.py      sonic clusters in full 512-d (UMAP->HDBSCAN)   -> clusters.sqlite
  4. umap_music.py   project those vectors to a 2D+3D map (PaCMAP)  -> umap.sqlite
  5. umap_artists.py artist-level map (UMAP)                        -> artist_umap.sqlite
  6. waveform.py     colored 3-band DJ waveforms                    -> waveforms.sqlite

Each writes a sidecar into <root>/.crate/ that the Crate app reads. Point it at your music with
--root, or set CRATE_LIB_ROOT, or it falls back to lib_root in crate_config.json (set by the app's
⚙ FOLDERS). Idempotent: re-running only processes new/changed tracks (--rebuild forces all).

Usage:  python analyze_all.py [--root "D:/Music"] [--limit N] [--rebuild] [--skip embed_muq.py]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# (label, script, which flags it accepts)
STEPS = [
    ("BPM / key / energy", "analyze.py", {"limit", "rebuild"}),
    ("MuQ-MuLan embeddings", "embed_muq.py", {"limit", "rebuild"}),
    ("sonic clusters (full 512-d)", "cluster.py", set()),
    ("PaCMAP map (tracks, 2D+3D)", "umap_music.py", set()),
    ("UMAP map (artists)", "umap_artists.py", set()),
    ("colored waveforms", "waveform.py", {"limit", "rebuild"}),
]


def main():
    ap = argparse.ArgumentParser(description="Run Crate's local analysis pipeline")
    ap.add_argument("--root", default=None, help="music library root (else CRATE_LIB_ROOT / config)")
    ap.add_argument("--limit", type=int, default=0, help="cap tracks per step (for a quick test)")
    ap.add_argument("--rebuild", action="store_true", help="re-analyze everything, not just new")
    ap.add_argument("--skip", default="", help="comma list of step scripts to skip")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    frozen = getattr(sys, "frozen", False)      # running inside the bundled crate-analyze exe?
    for label, script, supports in STEPS:
        if script in skip:
            print(f"\n----- skipping {script} -----", flush=True)
            continue
        if frozen:
            # re-invoke our own exe with --step <module>; there is no <step>.py on disk to run
            cmd = [sys.executable, "--step", Path(script).stem]
        else:
            cmd = [sys.executable, str(HERE / script)]
        if args.root:
            cmd += ["--root", args.root]
        if args.limit and "limit" in supports:
            cmd += ["--limit", str(args.limit)]
        if args.rebuild and "rebuild" in supports:
            cmd += ["--rebuild"]
        print(f"\n===== {label}  ({script}) =====", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!! {script} failed (exit {r.returncode}); stopping.", flush=True)
            return r.returncode
        print(f"----- {label} done in {time.time() - t0:.0f}s -----", flush=True)

    print("\n=== analysis complete — open Crate: BPM/key columns, the MAP, and colored "
          "waveforms are now populated ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
