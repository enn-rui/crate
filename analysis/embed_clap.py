#!/usr/bin/env python3
"""embed_clap.py — CLAP audio embeddings for Crate, run ON THE BOX (offloaded, CPU).

Embeds each owned track into a 512-d CLAP semantic vector (laion/clap-htsat-unfused via
HF transformers) and stores it in <lib_root>/.crate/music_vectors.sqlite. Feed that to
umap_music.py to get 2D coords for the Crate MAP view, and (optionally) to a Qdrant
music_audio collection for text->music semantic search via the embedding hub.

The whole-track vector averages `--clips` (default 6) evenly-spaced 10s windows — more
representative than one mid clip. It also stores per-section intro/outro vectors (one 10s window
near each end) for transition/mix-point matching (outro-of-A vs intro-of-B). Idempotent by
relpath+mtime; runs on GPU when present (e.g. a 5090), else CPU.

Usage:  ~/crate/.venv/bin/python embed_clap.py [--root DIR] [--clips 6] [--no-sections] [--limit N] [--rebuild]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from _common import resolve_root, crate_dir, iter_audio, parse_buckets  # local library ROOT + sidecar dir

MODEL = "laion/larger_clap_music"   # CLAP fine-tuned on MUSIC — better DJ-track clustering than the general model
SR = 48000
CLIP_SECONDS = 10.0
DEFAULT_CLIPS = 6   # whole-track average is sampled from this many evenly-spaced windows (was 3)


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute(
        """CREATE TABLE IF NOT EXISTS vectors (
            relpath TEXT PRIMARY KEY,
            mtime   REAL,
            dim     INTEGER,
            vec     BLOB,             -- float32 little-endian, length dim (whole-track average)
            added_at REAL
        )"""
    )
    # Per-section vectors (intro / outro) for future mix-point matching: a track's OUTRO vector
    # vs a candidate's INTRO vector says how well the *transition* fits, not just the whole vibe.
    # Nullable + added by migration so old DBs upgrade in place. body == whole-track `vec`.
    cols = {r[1] for r in con.execute("PRAGMA table_info(vectors)")}
    for c in ("vec_intro", "vec_outro"):
        if c not in cols:
            con.execute(f"ALTER TABLE vectors ADD COLUMN {c} BLOB")
    con.commit()
    return con


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="music library root (else CRATE_LIB_ROOT / config)")
    ap.add_argument("--buckets", default=None,
                    help="only these top-level folders (CSV), e.g. music,dj,music-mp3 — skips "
                         "download/quarantine dirs sharing the root")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--clips", type=int, default=DEFAULT_CLIPS,
                    help="windows averaged for the whole-track vector (more = more representative)")
    ap.add_argument("--no-sections", action="store_true",
                    help="skip the intro/outro per-section vectors (whole-track average only)")
    ap.add_argument("--stats-only", action="store_true",
                    help="don't embed; just recompute + store the dataset-mean (vector_stats) over "
                         "the existing vectors. Fast (no model load) — run after pruning/trimming.")
    args = ap.parse_args(argv)
    root = resolve_root(args.root)

    import vectors as vec_stats   # shared embedding transform (dataset-mean persistence)

    # Fast path: recompute the centering mean from whatever's already in the table and exit.
    if args.stats_only:
        con = connect(crate_dir(root) / "music_vectors.sqlite")
        m = vec_stats.recompute_and_store(con)
        n = con.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        con.close()
        print(f"=== vector_stats: stored dataset-mean over {n} vectors "
              f"({'ok' if m is not None else 'NO VECTORS'}) ===", flush=True)
        return 0

    import numpy as np
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"   # use the GPU when present (e.g. a 5090)
    print(f"loading {MODEL} on {device}...", flush=True)
    model = ClapModel.from_pretrained(MODEL).to(device)
    proc = ClapProcessor.from_pretrained(MODEL)
    model.eval()

    n_clips = max(1, args.clips)
    want_sections = not args.no_sections

    def load_clip(path, off):
        y, _ = librosa.load(str(path), sr=SR, mono=True, offset=max(0.0, off), duration=CLIP_SECONDS)
        return y if y.size >= SR else None  # need >=1s of audio in the window

    buckets = parse_buckets(args.buckets)
    con = connect(crate_dir(root) / "music_vectors.sqlite")
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
            dur = librosa.get_duration(path=str(p))
            long_enough = dur > CLIP_SECONDS * 1.6
            # WHOLE-TRACK: n_clips evenly-spaced windows (more = better coverage of intro/drop/outro).
            if long_enough:
                span = dur - CLIP_SECONDS
                fracs = [(i + 0.5) / n_clips for i in range(n_clips)]  # centers, e.g. 6 -> .08..0.92
                body_offs = [span * f for f in fracs]
            else:
                body_offs = [0.0]
            body = [c for c in (load_clip(p, off) for off in body_offs) if c is not None]
            if not body:
                raise ValueError("too short")
            # SECTIONS: a dedicated intro window (near the start) and outro window (near the end),
            # for transition/mix-point matching. Skipped on short tracks (intro==outro==whole).
            labels = ["body"] * len(body)
            clips = list(body)
            if want_sections and long_enough:
                intro = load_clip(p, min(8.0, span))               # ~8s in, past dead-air
                outro = load_clip(p, max(0.0, dur - CLIP_SECONDS - 8.0))
                if intro is not None:
                    clips.append(intro); labels.append("intro")
                if outro is not None:
                    clips.append(outro); labels.append("outro")
            # batch ALL clips of this track through CLAP in ONE forward pass
            inputs = proc(audio=clips, sampling_rate=SR, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = model.get_audio_features(**inputs)
            # transformers 5.x returns a model-output object whose projected joint-space embedding
            # is pooler_output; older/other paths return the tensor directly. Shape: (n_clips, 512).
            out = getattr(feats, "pooler_output", feats).cpu().numpy().astype("float32")

            def norm(v):
                return (v / (np.linalg.norm(v) + 1e-9)).astype("float32")

            # Whole-track average: L2-normalize EACH clip first, then average, then renorm. Without
            # the per-clip norm a single loud window dominates the mean; and averaging un-normed
            # CLAP features also pulls every track toward the global mean direction, worsening the
            # anisotropy that vectors.center_l2 then has to undo. (Existing DBs keep their old
            # vectors until a full --rebuild; centering dominates either way.)
            body_rows = out[[i for i, l in enumerate(labels) if l == "body"]]
            body_unit = body_rows / (np.linalg.norm(body_rows, axis=1, keepdims=True) + 1e-9)
            vec = norm(body_unit.mean(axis=0))                     # whole-track average
            intro_b = outro_b = None
            for i, l in enumerate(labels):
                if l == "intro":
                    intro_b = norm(out[i]).tobytes()
                elif l == "outro":
                    outro_b = norm(out[i]).tobytes()
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
    vec_stats.recompute_and_store(con)   # refresh the centering mean over the full (updated) set
    con.close()
    print(f"\n=== embedded {ok} new, {skip} skipped, {err} errors in {time.time()-t0:.0f}s; "
          f"music_vectors.sqlite holds {total} (vector_stats refreshed) ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
