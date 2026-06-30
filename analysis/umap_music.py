#!/usr/bin/env python3
"""umap_music.py — project the music embeddings to 2D+3D for the Crate MAP view.

Reads <root>/.crate/music_vectors.sqlite (mean-centered like the app), reduces with PaCMAP
for good global structure, normalizes to [0,1], and writes <root>/.crate/umap.sqlite
(coords: relpath,x,y  +  coords3d: relpath,x,y,z) which the Crate app reads.

Root resolves via --root / CRATE_LIB_ROOT / crate_config.json (see _common.resolve_root).
Usage:  python umap_music.py [--root "D:/Music"] [--neighbors 15] [--min-dist 0.1]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from _common import resolve_root, crate_dir  # local library ROOT + sidecar dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="music library root (else CRATE_LIB_ROOT / config)")
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    args = ap.parse_args(argv)
    root = resolve_root(args.root)
    VEC_DB = crate_dir(root) / "music_vectors.sqlite"
    OUT_DB = crate_dir(root) / "umap.sqlite"

    import vectors as vec_stats

    con = sqlite3.connect(str(VEC_DB))
    rows = con.execute("SELECT relpath, dim, vec FROM vectors").fetchall()
    mean = vec_stats.load_mean(con)   # persisted dataset-mean (None until embed writes vector_stats)
    con.close()
    if not rows:
        sys.exit("no vectors in music_vectors.sqlite — run embed_muq.py first")
    relpaths = [r[0] for r in rows]
    mat = np.stack([np.frombuffer(r[2], dtype=np.float32, count=r[1]) for r in rows])
    # MEAN-CENTER + renorm before projecting — audio embeddings carry a strong common-mode
    # direction (everything looks similar), so a fit on raw vectors is noise. center_l2 removes it.
    mat = vec_stats.center_l2(mat, mean)
    if len(relpaths) < 5:
        # Too few tracks for a meaningful neighbor graph (UMAP needs a handful to be stable, and
        # n_neighbors collapses to 0 at len==1). Write trivial coords and SUCCEED so a fresh/small
        # library — or an agent test-running the pipeline — doesn't see the whole pass fail.
        import math
        n = len(relpaths)
        pts = ([(0.5, 0.5)] if n == 1 else
               [(0.5 + 0.25 * math.cos(2 * math.pi * i / n),
                 0.5 + 0.25 * math.sin(2 * math.pi * i / n)) for i in range(n)])
        out = sqlite3.connect(str(OUT_DB))
        out.execute("CREATE TABLE IF NOT EXISTS coords (relpath TEXT PRIMARY KEY, x REAL, y REAL)")
        out.execute("DELETE FROM coords")
        out.executemany("INSERT INTO coords(relpath,x,y) VALUES(?,?,?)",
                        [(relpaths[i], float(pts[i][0]), float(pts[i][1])) for i in range(n)])
        out.commit()
        out.close()
        print(f"=== only {n} tracks — wrote trivial coords (UMAP needs >=5) ===", flush=True)
        return 0
    print(f"PaCMAP on {mat.shape[0]} vectors x {mat.shape[1]}d "
          f"(transform={'center' if mean is not None else 'RAW(no stats!)'})...", flush=True)

    # PaCMAP instead of UMAP: it preserves GLOBAL structure far better (distinct genres stay apart
    # instead of being crushed together), which is the main cause of "unrelated song next to a good
    # cluster" in the 2D view. Euclidean on the mean-centered unit vectors ≈ cosine. We emit BOTH a
    # 2D layout (what the current map reads) and a 3D layout (for the 3D map view) from the SAME fit
    # settings. NOTE: cluster membership is computed separately in full 512-d (cluster.py) — never
    # read clusters off this 2D/3D projection.
    import pacmap
    matf = mat.astype("float32")

    def _project(dim):
        r = pacmap.PaCMAP(n_components=dim, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0,
                          random_state=42)
        z = r.fit_transform(matf, init="pca").astype("float32")
        # Normalize into [0,1] with a SINGLE uniform scale (not per-axis), so the relative geometry
        # is preserved — per-axis min-max would stretch axes independently and distort nearness.
        lo, hi = z.min(0), z.max(0)
        span = float((hi - lo).max()) + 1e-9
        z = (z - lo) / span
        z = z + (1.0 - (z.max(0) + z.min(0))) / 2.0   # center each axis within [0,1]
        return z

    t0 = time.time()
    xy = _project(2)
    xyz = _project(3)   # 3D layout for the (future) 3D map view; harmless/unused by the 2D view

    out = sqlite3.connect(str(OUT_DB))
    out.execute("CREATE TABLE IF NOT EXISTS coords (relpath TEXT PRIMARY KEY, x REAL, y REAL)")
    out.execute("CREATE TABLE IF NOT EXISTS coords3d "
                "(relpath TEXT PRIMARY KEY, x REAL, y REAL, z REAL)")
    out.execute("DELETE FROM coords")
    out.execute("DELETE FROM coords3d")
    out.executemany("INSERT INTO coords(relpath,x,y) VALUES(?,?,?)",
                    [(relpaths[i], float(xy[i, 0]), float(xy[i, 1])) for i in range(len(relpaths))])
    out.executemany("INSERT INTO coords3d(relpath,x,y,z) VALUES(?,?,?,?)",
                    [(relpaths[i], float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
                     for i in range(len(relpaths))])
    out.commit()
    out.close()
    print(f"=== wrote {len(relpaths)} coords (2D+3D, PaCMAP) to {OUT_DB} in "
          f"{time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
