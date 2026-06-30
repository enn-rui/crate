#!/usr/bin/env python3
"""waveform.py — colored DJ waveforms for Crate, run ON THE BOX (offloaded).

Walks the library root, decodes each FULL track, and computes a downsampled
3-band (low / mid / high) waveform: N bins, each holding the low / mid / high energy at that
point in the song (uint8, 0-255). The PC app reads this sidecar over Z: and renders a
rekordbox-style colored waveform with a playhead + cue markers.

Sidecar: <lib_root>/.crate/waveforms.sqlite, keyed by relpath (e.g. music/Artist/Title.flac).
Idempotent: a track is skipped if relpath+mtime already match (use --rebuild to force).

Usage: ~/crate/.venv/bin/python waveform.py [--buckets music,dj,music-mp3] [--bins 1600] [--limit N] [--rebuild]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from _common import resolve_root, crate_dir, iter_audio  # local library ROOT + sidecar dir

SR = 22050          # plenty for a waveform shape; Nyquist 11 kHz covers the visible band split
NFFT = 2048
HOP = 512
LOW_HZ = 200.0      # < low  = bass band
MID_HZ = 2000.0     # low..mid = mids ; > mid = highs


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute(
        """CREATE TABLE IF NOT EXISTS waveforms (
            relpath TEXT PRIMARY KEY,   -- music/Artist/Title.flac (forward slashes)
            mtime   REAL,
            bins    INTEGER,            -- number of columns
            data    BLOB,               -- bins*3 uint8, interleaved lo,mid,hi per column
            analyzed_at REAL
        )"""
    )
    con.commit()
    return con


def compute_waveform(path: Path, bins: int) -> bytes:
    import librosa
    import numpy as np

    y, sr = librosa.load(str(path), sr=SR, mono=True)
    if y.size < sr // 2:
        raise ValueError("too short / unreadable")
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))      # (1+NFFT/2, frames)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    lo = S[freqs < LOW_HZ].sum(0)
    mid = S[(freqs >= LOW_HZ) & (freqs < MID_HZ)].sum(0)
    hi = S[freqs >= MID_HZ].sum(0)
    frames = lo.shape[0]
    if frames == 0:
        raise ValueError("no frames")

    # downsample the time axis to `bins` columns (max within each chunk -> punchy peaks)
    idx = np.linspace(0, frames, bins + 1).astype(int)
    out = np.zeros((bins, 3), dtype=np.float64)
    for b, (a, c) in enumerate(zip(idx[:-1], idx[1:])):
        c = max(c, a + 1)
        out[b, 0] = lo[a:c].max()
        out[b, 1] = mid[a:c].max()
        out[b, 2] = hi[a:c].max()
    # normalize by the global peak across all bands so the band BALANCE is preserved (colour),
    # then gamma-lift a touch so quiet detail is visible, and quantize to uint8.
    peak = out.max()
    if peak <= 0:
        raise ValueError("silent")
    norm = (out / peak) ** 0.7
    u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return u8.tobytes()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="music library root (else CRATE_LIB_ROOT / config)")
    ap.add_argument("--bins", type=int, default=1600)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    root = resolve_root(args.root)

    con = connect(crate_dir(root) / "waveforms.sqlite")
    have = {r[0]: r[1] for r in con.execute("SELECT relpath, mtime FROM waveforms")}
    done = ok = skip = err = 0
    t0 = time.time()
    for p in iter_audio(root):
        rel = p.relative_to(root).as_posix()
        mt = p.stat().st_mtime
        if not args.rebuild and rel in have and abs(have[rel] - mt) < 1.0:
            skip += 1
            continue
        try:
            data = compute_waveform(p, args.bins)
            con.execute(
                """INSERT INTO waveforms (relpath,mtime,bins,data,analyzed_at)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(relpath) DO UPDATE SET
                     mtime=excluded.mtime, bins=excluded.bins, data=excluded.data,
                     analyzed_at=excluded.analyzed_at""",
                (rel, mt, args.bins, data, time.time()),
            )
            con.commit()
            ok += 1
            print(f"OK  {rel}", flush=True)
        except Exception as e:
            err += 1
            print(f"ERR {rel}: {type(e).__name__}: {e}", flush=True)
        done += 1
        if args.limit and done >= args.limit:
            break
    total = con.execute("SELECT COUNT(*) FROM waveforms").fetchone()[0]
    con.close()
    print(f"\n=== waveforms: {ok} new, {skip} skipped, {err} errors in "
          f"{time.time() - t0:.0f}s; sidecar now holds {total} ===")


if __name__ == "__main__":
    sys.exit(main())
