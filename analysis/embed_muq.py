#!/usr/bin/env python3
"""embed_muq.py — MuQ-MuLan audio embeddings with SALIENT-SEGMENT pooling (run ON THE BOX).

Replaces embed_clap.py's two problems at once:
  1. MODEL: MuQ-MuLan (OpenMuQ/MuQ-MuLan-large, 512-d) instead of CLAP. MuQ is music-native and far
     less anisotropic out of the box (measured raw cosine 0.63 between two random tracks vs CLAP's
     0.985), so its similarity actually tracks genre/groove, not just timbre/production.
  2. POOLING: instead of averaging 6 evenly-spaced windows (which blends a quiet intro + a hard drop
     into a muddy midpoint vector that represents NO real moment of the song), we find each track's
     most energetic ~SALIENT_SECONDS window — its "drop"/core — and embed THAT as the track's
     identity. Intro/outro windows are embedded separately (vec_intro/vec_outro) for transition match.

Writes 512-d float32 vectors to <root>/.crate/<out> (default music_vectors.sqlite — same schema as
embed_clap.py, so cluster.py / umap_music.py / the app all consume it unchanged). Idempotent by
relpath+mtime. 24 kHz mono, fp32 (MuQ recommends fp32 to avoid NaNs). GPU if present, else CPU.

Usage:  python embed_muq.py [--root DIR] [--limit N] [--rebuild] [--out NAME] [--salient 30]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from _common import resolve_root, crate_dir, iter_audio, parse_buckets

MODEL = "OpenMuQ/MuQ-MuLan-large"
SR = 24000
SALIENT_SECONDS = 30.0   # length of the "core/drop" window embedded as the track's identity
END_SECONDS = 12.0       # intro/outro window length (for transition matching)


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute(
        """CREATE TABLE IF NOT EXISTS vectors (
            relpath TEXT PRIMARY KEY, mtime REAL, dim INTEGER, vec BLOB, added_at REAL)"""
    )
    cols = {r[1] for r in con.execute("PRAGMA table_info(vectors)")}
    for c in ("vec_intro", "vec_outro"):
        if c not in cols:
            con.execute(f"ALTER TABLE vectors ADD COLUMN {c} BLOB")
    con.commit()
    return con


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--buckets", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="music_vectors.sqlite",
                    help="output sidecar name under <root>/.crate/ (use a test name to avoid "
                         "overwriting the live CLAP vectors)")
    ap.add_argument("--salient", type=float, default=SALIENT_SECONDS)
    ap.add_argument("--no-sections", action="store_true",
                    help="skip the intro/outro window embeds (3x fewer forwards) — only the salient "
                         "whole-track vector, which feeds the map/clusters. Add sections later if the "
                         "transition-scoring feature is wanted.")
    args = ap.parse_args(argv)
    root = resolve_root(args.root)

    import numpy as np
    import librosa
    import torch
    from muq import MuQMuLan
    import vectors as vec_stats

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL} on {device}...", flush=True)
    model = MuQMuLan.from_pretrained(MODEL).to(device).eval()

    win = int(args.salient * SR)
    end_win = int(END_SECONDS * SR)

    def embed(seg):
        """512-d unit vector for one mono 24k segment."""
        with torch.no_grad():
            z = model(wavs=torch.tensor(seg).unsqueeze(0).float().to(device))
        v = z.squeeze(0).float().cpu().numpy().astype("float32")
        return (v / (np.linalg.norm(v) + 1e-9)).astype("float32")

    def salient_window(y):
        """The most energetic `win`-sample stretch of the track (its drop/core)."""
        if len(y) <= win:
            return y
        # 1-second RMS envelope, then the W-second window with the highest summed energy.
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=SR)[0]
        W = max(1, int(args.salient))
        if len(rms) <= W:
            return y[:win]
        csum = np.cumsum(np.insert(rms, 0, 0))
        wins = csum[W:] - csum[:-W]
        s = int(np.argmax(wins)) * SR
        return y[s:s + win]

    buckets = parse_buckets(args.buckets)
    con = connect(crate_dir(root) / args.out)
    have = {r[0]: r[1] for r in con.execute("SELECT relpath, mtime FROM vectors")}
    ok = skip = err = 0
    t0 = time.time()
    for p in iter_audio(root, buckets):
        rel = p.relative_to(root).as_posix()
        mt = p.stat().st_mtime
        if not args.rebuild and rel in have and abs(have[rel] - mt) < 1.0:
            skip += 1
            continue
        try:
            y, _ = librosa.load(str(p), sr=SR, mono=True)
            if y.size < SR:
                raise ValueError("too short")
            vec = embed(salient_window(y))
            intro_b = outro_b = None
            if not args.no_sections and y.size > end_win * 2 + SR:
                intro_b = embed(y[:end_win]).tobytes()
                outro_b = embed(y[-end_win:]).tobytes()
            con.execute(
                """INSERT INTO vectors(relpath,mtime,dim,vec,vec_intro,vec_outro,added_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(relpath) DO UPDATE SET
                     mtime=excluded.mtime, dim=excluded.dim, vec=excluded.vec,
                     vec_intro=excluded.vec_intro, vec_outro=excluded.vec_outro,
                     added_at=excluded.added_at""",
                (rel, mt, int(vec.shape[0]), vec.tobytes(), intro_b, outro_b, time.time()))
            con.commit()
            ok += 1
            if ok % 10 == 0:
                print(f"  {ok} embedded ({(time.time()-t0)/ok:.1f}s/track) … {rel}", flush=True)
        except Exception as e:
            err += 1
            print(f"ERR {rel}: {type(e).__name__}: {e}", flush=True)
        if args.limit and (ok + err) >= args.limit:
            break
    total = con.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    vec_stats.recompute_and_store(con)
    con.close()
    print(f"\n=== MuQ embedded {ok} new, {skip} skipped, {err} errors in {time.time()-t0:.0f}s; "
          f"{args.out} holds {total} (vector_stats refreshed) ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
