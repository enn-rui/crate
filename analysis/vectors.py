#!/usr/bin/env python3
"""vectors.py — the embedding transform shared by every consumer of the sonic vectors.

Raw audio embeddings can be badly *anisotropic* (the "cone effect"): every track is highly
cosine-similar to every other, so nearest-neighbour is noise and BOTH the 512-d similarity
(mixability/similar_tracks) AND the maps degrade to mush. The fix is to **mean-center** the
vectors (subtract the dataset-mean vector) and re-L2-normalize before any cosine/projection — that
removes the common-mode component and restores genre structure (measured on the original CLAP
model: off-diagonal cosine +0.985 -> ~0.00, dance-genre separation +0.169 vs +0.066 for PCA-drop,
which over-flattens; MuQ-MuLan is far less anisotropic but centering still helps and is kept).

The dataset-mean is computed ONCE over the whole embedded library and persisted INSIDE
music_vectors.sqlite (a `vector_stats` table), so the box (UMAP build) and the PC (similarity
at read time) apply the *identical* transform and never disagree about who is near whom.

`library.py` re-implements `center_l2` inline (it can't import this analysis/ module from the app
venv); keep the two in sync. The transform is intentionally simple and stable.
"""
from __future__ import annotations

import sqlite3

import numpy as np

TRANSFORM = "center"   # persisted alongside the mean; bump if the transform ever changes


def _all_vectors(con: sqlite3.Connection) -> np.ndarray:
    """Every whole-track vector in the table, as float64 (N, dim). Empty -> (0,0)."""
    rows = con.execute("SELECT dim, vec FROM vectors WHERE vec IS NOT NULL").fetchall()
    if not rows:
        return np.zeros((0, 0))
    return np.stack([np.frombuffer(v, dtype=np.float32, count=d) for d, v in rows]).astype(np.float64)


def recompute_and_store(con: sqlite3.Connection) -> np.ndarray | None:
    """Compute the dataset-mean over ALL embedded tracks and store it (+ the transform name) in a
    `vector_stats` table inside the same db. Returns the mean (float32) or None if no vectors.
    Call this at the END of an embed run (after new tracks land) so the mean stays current."""
    M = _all_vectors(con)
    con.execute("CREATE TABLE IF NOT EXISTS vector_stats (k TEXT PRIMARY KEY, v BLOB)")
    if M.shape[0] == 0:
        con.commit()
        return None
    mean = M.mean(axis=0).astype(np.float32)
    con.execute("INSERT OR REPLACE INTO vector_stats(k, v) VALUES('mean', ?)", (mean.tobytes(),))
    con.execute("INSERT OR REPLACE INTO vector_stats(k, v) VALUES('transform', ?)",
                (TRANSFORM.encode(),))
    con.execute("INSERT OR REPLACE INTO vector_stats(k, v) VALUES('count', ?)",
                (str(M.shape[0]).encode(),))
    con.commit()
    return mean


def load_mean(con: sqlite3.Connection) -> np.ndarray | None:
    """The persisted dataset-mean (float32), or None if vector_stats isn't present yet."""
    try:
        r = con.execute("SELECT v FROM vector_stats WHERE k='mean'").fetchone()
    except sqlite3.OperationalError:
        return None
    if not r or not r[0]:
        return None
    return np.frombuffer(r[0], dtype=np.float32)


def center_l2(M: np.ndarray, mean: np.ndarray | None) -> np.ndarray:
    """Subtract the dataset-mean and re-L2-normalize (rows). If `mean` is None, just L2-normalize
    (back-compat: behaves like the old raw-cosine path). Accepts (N, d) or (d,)."""
    X = np.atleast_2d(M).astype(np.float64)
    if mean is not None:
        X = X - mean.astype(np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    out = X / (n + 1e-12)
    return out.reshape(M.shape) if M.ndim == 1 else out
