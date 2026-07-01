#!/usr/bin/env python3
"""umap_artists.py — ARTIST-level UMAP for Crate's artist map.

Averages each artist's per-track sonic vectors (from music_vectors.sqlite) into one "artist vector",
then UMAPs those artist vectors so sonically similar artists sit near each other. Writes
artist_umap.sqlite (artist, x, y, n) — x,y in [0,1], n = how many of the artist's tracks fed in.

Artist is taken from the library's filing convention (<bucket>/<Artist>/<Title.ext> -> the parent
folder). Run after embed_muq.py. Usage: python umap_artists.py [--root <music folder>] [--min-tracks 1]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _common import resolve_root, crate_dir  # local library ROOT + sidecar dir


def primary_artist(name: str) -> str:
    """First-billed artist: split on collab separators and trim, so 'A; B' / 'A feat. B' -> 'A'."""
    if not name:
        return ""
    first = re.split(r"\s*(?:;|,|/|&|\bfeat\.?\b|\bft\.?\b|\bx\b|\bvs\.?\b)\s*", name, flags=re.I)[0]
    return first.strip()


def folder_artist(relpath: str) -> str:
    parts = relpath.split("/")
    return parts[-2] if len(parts) >= 2 else "?"


def build_resolver(root, artist_map_path):
    """relpath -> artist. Prefer real tags (an --artist-map json, else the app's crate.db); fall
    back to the folder name. When real tags are available the result is RESTRICTED to those tracks
    (the app's actual library), dropping box-only junk and album-folder pseudo-artists."""
    if artist_map_path and Path(artist_map_path).exists():
        m = json.loads(Path(artist_map_path).read_text(encoding="utf-8"))
        # clean the first-billed artist even from a raw {relpath: tag} map, so collabs like
        # 'Hugel, Solto (Fr)' collapse to 'Hugel' instead of becoming their own pseudo-artist
        return (lambda rel: primary_artist(m[rel]) if m.get(rel) else None), True
    # Pull real artist tags straight from the app's crate.db (no `import library` — that needs the
    # app venv + mutagen and isn't importable from this analysis/ subdir). crate.db lives next to
    # the app (../crate.db); override with CRATE_DB. Falls back to folder names if it isn't there.
    crate_db = Path(os.environ.get("CRATE_DB", Path(__file__).resolve().parent.parent / "crate.db"))
    try:
        if crate_db.exists():
            con = sqlite3.connect(str(crate_db))
            bypath = {r[0]: r[1] for r in con.execute("SELECT path, artist FROM tracks")}
            con.close()
            if bypath:
                def res(rel):
                    a = bypath.get(str(Path(root) / rel))
                    return primary_artist(a) if a else None
                return res, True
    except Exception:
        pass
    return (lambda rel: folder_artist(rel)), False      # folder fallback, keep everything


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="music library root (else CRATE_LIB_ROOT / config)")
    # tight local structure: low neighbors + min_dist=0 packs sonically-similar artists into
    # visible clumps instead of an even spread across the canvas (the coords are min-max
    # normalized to fill [0,1], so a loose min_dist reads as "scattered, no clusters").
    ap.add_argument("--neighbors", type=int, default=8)
    ap.add_argument("--min-dist", type=float, default=0.0)
    ap.add_argument("--min-tracks", type=int, default=1, help="drop artists with fewer tracks")
    ap.add_argument("--artist-map", default=None, help="json {relpath: artist} of real tags")
    args = ap.parse_args(argv)
    root = resolve_root(args.root)
    VEC_DB = crate_dir(root) / "music_vectors.sqlite"
    OUT_DB = crate_dir(root) / "artist_umap.sqlite"

    import vectors as vec_stats

    con = sqlite3.connect(str(VEC_DB))
    rows = con.execute("SELECT relpath, dim, vec FROM vectors").fetchall()
    mean = vec_stats.load_mean(con)
    con.close()
    if not rows:
        sys.exit("no vectors in music_vectors.sqlite — run embed_muq.py first")
    print(f"transform={'center' if mean is not None else 'RAW(no stats!)'}", flush=True)

    resolve, restricted = build_resolver(root, args.artist_map)
    print(f"artist source: {'real tags (restricted to library)' if restricted else 'folder names'}",
          flush=True)
    # group track vectors by artist (case-insensitive so 'Charli XCX' == 'Charli xcx'),
    # average -> one L2-normalized vector per artist; label with the most common casing.
    by_artist: dict[str, list] = defaultdict(list)
    casing: dict[str, Counter] = defaultdict(Counter)
    for rel, dim, vec in rows:
        art = resolve(rel)
        if not art or art == "?":
            continue                          # unresolved / box-only track -> skip
        key = art.lower()
        # MEAN-CENTER each TRACK vector BEFORE the per-artist average — centering has to happen at
        # the track level; averaging first would already have collapsed the spread we're restoring.
        v = vec_stats.center_l2(np.frombuffer(vec, dtype=np.float32, count=dim), mean)
        by_artist[key].append(v)
        casing[key][art] += 1
    artists, mats = [], []
    for key, vecs in by_artist.items():
        if len(vecs) >= args.min_tracks:
            v = np.mean(vecs, axis=0)
            v /= (np.linalg.norm(v) + 1e-9)
            artists.append((casing[key].most_common(1)[0][0], len(vecs)))
            mats.append(v)
    if len(mats) < 3:
        # Too few artists for a meaningful map (a fresh/small library, or an agent test-running
        # the pipeline). Write an empty table and SUCCEED — don't fail the whole analysis pass.
        out = sqlite3.connect(str(OUT_DB))
        out.execute("CREATE TABLE IF NOT EXISTS artists (artist TEXT PRIMARY KEY, x REAL, y REAL, n INTEGER)")
        out.execute("DELETE FROM artists")
        out.execute("CREATE TABLE IF NOT EXISTS artists3d "
                    "(artist TEXT PRIMARY KEY, x REAL, y REAL, z REAL, n INTEGER)")
        out.execute("DELETE FROM artists3d")
        out.commit()
        out.close()
        print(f"=== only {len(mats)} artists meet --min-tracks {args.min_tracks}; "
              f"skipped artist map (need >=3) ===", flush=True)
        return 0
    mat = np.stack(mats)
    print(f"UMAP on {len(artists)} artists (from {len(rows)} tracks)...", flush=True)

    import umap

    def _fit(ncomp):
        # UMAP's default SPECTRAL init needs N > n_components+1 (it asks scipy.linalg.eigh for k
        # eigenvectors of an N-node graph and fails when k >= N), which blows up on small libraries.
        # Use random init below a safe size so a few-artist library still maps instead of crashing.
        n = len(artists)
        init = "spectral" if n > max(ncomp + 2, 11) else "random"
        r = umap.UMAP(n_components=ncomp, n_neighbors=min(args.neighbors, n - 1),
                      min_dist=args.min_dist, metric="cosine", init=init, random_state=42)
        c = r.fit_transform(mat).astype("float32")
        lo, hi = c.min(0), c.max(0)
        span = float((hi - lo).max()) + 1e-9          # uniform scale into [0,1] (preserve UMAP shape)
        c = (c - lo) / span
        return c + (1.0 - (c.max(0) + c.min(0))) / 2.0

    t0 = time.time()
    try:
        xy = _fit(2)
        xyz = _fit(3)                                  # 3D variant for the orbitable artist galaxy
    except Exception as e:
        # never let a projection edge case fail the whole analysis pass — skip the artist map cleanly
        out = sqlite3.connect(str(OUT_DB))
        out.execute("CREATE TABLE IF NOT EXISTS artists (artist TEXT PRIMARY KEY, x REAL, y REAL, n INTEGER)")
        out.execute("DELETE FROM artists")
        out.execute("CREATE TABLE IF NOT EXISTS artists3d "
                    "(artist TEXT PRIMARY KEY, x REAL, y REAL, z REAL, n INTEGER)")
        out.execute("DELETE FROM artists3d")
        out.commit(); out.close()
        print(f"=== artist UMAP skipped ({type(e).__name__}: {e}) — too few/degenerate artists ===",
              flush=True)
        return 0

    out = sqlite3.connect(str(OUT_DB))
    out.execute("CREATE TABLE IF NOT EXISTS artists (artist TEXT PRIMARY KEY, x REAL, y REAL, n INTEGER)")
    out.execute("DELETE FROM artists")
    out.executemany("INSERT INTO artists(artist,x,y,n) VALUES(?,?,?,?)",
                    [(artists[i][0], float(xy[i, 0]), float(xy[i, 1]), artists[i][1])
                     for i in range(len(artists))])
    out.execute("CREATE TABLE IF NOT EXISTS artists3d "
                "(artist TEXT PRIMARY KEY, x REAL, y REAL, z REAL, n INTEGER)")
    out.execute("DELETE FROM artists3d")
    out.executemany("INSERT INTO artists3d(artist,x,y,z,n) VALUES(?,?,?,?,?)",
                    [(artists[i][0], float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]), artists[i][1])
                     for i in range(len(artists))])
    out.commit()
    out.close()
    print(f"=== wrote {len(artists)} artist coords (2D+3D) to {OUT_DB} in {time.time()-t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
