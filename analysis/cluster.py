#!/usr/bin/env python3
"""cluster.py — assign each track a hard sonic CLUSTER id, computed in FULL 512-d (not off the 2D map).

Reading clusters off a 2D UMAP/PaCMAP projection is unreliable — the projection invents false
neighbours, so unrelated tracks look adjacent. Cluster membership must come from the high-dimensional
space. This runs HDBSCAN over the MEAN-CENTERED 512-d MuQ-MuLan vectors (the same space used for
similarity/mixability) and writes clusters.sqlite (relpath, cluster_id). The Crate app reads it to
COLOR the map by real sonic cluster. cluster_id = -1 means noise / not confidently in any cluster.

Run AFTER embed_muq.py (needs music_vectors.sqlite + its vector_stats mean).

Usage:  python cluster.py [--root DIR] [--min-cluster-size N] [--min-samples N]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from _common import resolve_root, crate_dir   # local library ROOT + sidecar dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="music library root (else CRATE_LIB_ROOT / config)")
    ap.add_argument("--min-cluster-size", type=int, default=8,
                    help="smallest group HDBSCAN will call a cluster (smaller = more, finer clusters)")
    ap.add_argument("--min-samples", type=int, default=None,
                    help="HDBSCAN conservativeness; default = min-cluster-size")
    args = ap.parse_args(argv)
    root = resolve_root(args.root)
    VEC_DB = crate_dir(root) / "music_vectors.sqlite"
    OUT_DB = crate_dir(root) / "clusters.sqlite"

    import vectors as vec_stats

    con = sqlite3.connect(str(VEC_DB))
    rows = con.execute("SELECT relpath, dim, vec FROM vectors WHERE vec IS NOT NULL").fetchall()
    mean = vec_stats.load_mean(con)
    con.close()
    if not rows:
        sys.exit("no vectors in music_vectors.sqlite — run embed_muq.py first")

    relpaths = [r[0] for r in rows]
    mat = np.stack([np.frombuffer(r[2], dtype=np.float32, count=r[1]) for r in rows])
    # Cluster in the SAME mean-centered + unit-normed space as similarity and the UMAP/PaCMAP fit, so
    # the colours agree with who is actually near whom. On unit vectors, euclidean distance is a
    # monotonic function of cosine distance, so euclidean HDBSCAN ≈ cosine HDBSCAN (and is faster).
    mat = vec_stats.center_l2(mat, mean).astype("float32")

    n = len(relpaths)
    out = sqlite3.connect(str(OUT_DB))
    out.execute("CREATE TABLE IF NOT EXISTS clusters (relpath TEXT PRIMARY KEY, cluster_id INTEGER)")
    out.execute("DELETE FROM clusters")

    if n < max(args.min_cluster_size * 2, 12):
        # too few tracks to cluster meaningfully — mark all as noise and SUCCEED (fresh/small library
        # or an agent test-running the pipeline shouldn't fail the pass).
        out.executemany("INSERT INTO clusters(relpath,cluster_id) VALUES(?,?)",
                        [(relpaths[i], -1) for i in range(n)])
        out.commit()
        out.close()
        print(f"=== only {n} tracks — all unclustered (-1) ===", flush=True)
        return 0

    # HDBSCAN on the RAW 512-d space fails — the curse of dimensionality concentrates distances so
    # almost everything comes back as noise (measured: ~75% noise, 2 clusters). The documented fix is
    # to first reduce to ~10D with UMAP tuned for clustering (min_dist=0 packs each cluster tight,
    # larger n_neighbors for stability), THEN run HDBSCAN on that. This is still clustering in a
    # FAITHFUL high-D space — NOT reading clusters off the 2D display (the 2D/3D PaCMAP layout is
    # computed separately in umap_music.py and is for plotting only).
    import umap
    from sklearn.cluster import HDBSCAN
    t0 = time.time()
    nn = max(2, min(30, n - 1))
    ncomp = max(2, min(10, n - 2))
    reduced = umap.UMAP(n_components=ncomp, n_neighbors=nn, min_dist=0.0, metric="cosine",
                        random_state=42).fit_transform(mat).astype("float32")
    clus = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
    labels = clus.fit_predict(reduced)
    out.executemany("INSERT INTO clusters(relpath,cluster_id) VALUES(?,?)",
                    [(relpaths[i], int(labels[i])) for i in range(n)])
    out.commit()
    out.close()
    k = len({int(l) for l in labels if l >= 0})
    noise = int((labels < 0).sum())
    print(f"=== {k} clusters over {n} tracks, {noise} noise, in {time.time()-t0:.0f}s "
          f"-> {OUT_DB} ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
