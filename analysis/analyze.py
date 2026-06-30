#!/usr/bin/env python3
"""analyze.py — audio-feature analysis for Crate, run ON THE BOX (offloaded).

Walks the library root, computes BPM / musical key (Camelot) / energy / timbre per
track with librosa, and writes them to <lib_root>/.crate/features.sqlite. The PC app reads
that sidecar over Z: and merges BPM/key/energy into its own index.

FLAC is read via soundfile (no system ffmpeg needed). Idempotent: a track is skipped if its
relpath+mtime is already analyzed (use --rebuild to force). Analyzes a ~90s middle excerpt
for speed — plenty for BPM/key/energy.

Usage:  ~/crate/.venv/bin/python analyze.py [--buckets music,dj] [--limit N] [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from _common import resolve_root, crate_dir, iter_audio, parse_buckets  # local library ROOT + sidecar dir

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles
MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
# Camelot wheel: (pitch, mode) -> code.  'B' codes = major, 'A' codes = minor.
CAMELOT = {
    ("C", "maj"): "8B", ("G", "maj"): "9B", ("D", "maj"): "10B", ("A", "maj"): "11B",
    ("E", "maj"): "12B", ("B", "maj"): "1B", ("F#", "maj"): "2B", ("C#", "maj"): "3B",
    ("G#", "maj"): "4B", ("D#", "maj"): "5B", ("A#", "maj"): "6B", ("F", "maj"): "7B",
    ("A", "min"): "8A", ("E", "min"): "9A", ("B", "min"): "10A", ("F#", "min"): "11A",
    ("C#", "min"): "12A", ("G#", "min"): "1A", ("D#", "min"): "2A", ("A#", "min"): "3A",
    ("F", "min"): "4A", ("C", "min"): "5A", ("G", "min"): "6A", ("D", "min"): "7A",
}


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute(
        """CREATE TABLE IF NOT EXISTS features (
            relpath TEXT PRIMARY KEY,   -- e.g. music/Artist/Title.flac (forward slashes)
            mtime   REAL,
            bpm     REAL,
            key_camelot TEXT,
            key_name    TEXT,
            energy  REAL,
            centroid REAL,
            mfcc    TEXT,               -- JSON list of 13 MFCC means
            analyzed_at REAL
        )"""
    )
    # added features (nullable, migrated in place so old sidecars upgrade):
    #   lufs         — integrated loudness (LUFS); a perceptual loudness, better than raw RMS energy
    #   danceability — 0..1 pulse-clarity proxy (steadiness/strength of the beat) from librosa
    cols = {r[1] for r in con.execute("PRAGMA table_info(features)")}
    for c, decl in (("lufs", "REAL"), ("danceability", "REAL")):
        if c not in cols:
            con.execute(f"ALTER TABLE features ADD COLUMN {c} {decl}")
    con.commit()
    return con


def detect_key(chroma_mean):
    import numpy as np
    best = (-2.0, "C", "maj")
    for i in range(12):
        rot = np.roll(np.arange(12), -i)
        cm = chroma_mean[rot]
        for mode, prof in (("maj", MAJ), ("min", MIN)):
            c = float(np.corrcoef(cm, prof)[0, 1])
            if c > best[0]:
                best = (c, PITCHES[i], mode)
    _, tonic, mode = best
    return CAMELOT.get((tonic, mode), "?"), f"{tonic} {'major' if mode == 'maj' else 'minor'}"


def pulse_clarity(onset_env, sr, hop=512, bpm_lo=70.0, bpm_hi=160.0) -> float:
    """0..1 danceability proxy = strength of the dominant beat periodicity. Autocorrelate the
    onset envelope, normalize by lag-0, and take the peak within a plausible tempo lag band: a
    steady four-on-the-floor scores high, ambient/rubato low. (Not Essentia-grade, but dep-free.)"""
    import numpy as np
    import librosa
    if onset_env is None or len(onset_env) < 4:
        return 0.0
    oe = np.asarray(onset_env, dtype=float)
    oe = oe - oe.mean()                  # remove DC: onset_strength is non-negative, so without
    if not np.any(oe):                   # this the autocorr is dominated by raw energy, not the beat
        return 0.0
    ac = librosa.autocorrelate(oe)
    if ac.size == 0 or ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    fps = sr / hop                                   # onset frames per second
    lo = int(round(fps * 60.0 / bpm_hi))             # smaller lag = faster tempo
    hi = int(round(fps * 60.0 / bpm_lo))
    lo, hi = max(1, lo), min(len(ac) - 1, hi)
    if hi <= lo:
        return 0.0
    return float(max(0.0, min(1.0, ac[lo:hi + 1].max())))


def integrated_lufs(y, sr) -> float | None:
    """Integrated loudness (LUFS) via pyloudnorm; None if the dep/measure is unavailable."""
    try:
        import pyloudnorm as pyln
        import numpy as np
        meter = pyln.Meter(sr)                        # BS.1770 K-weighting at this sample rate
        val = float(meter.integrated_loudness(np.asarray(y, dtype=float)))
        return None if (val != val or val == float("-inf")) else round(val, 2)
    except Exception:
        return None


def analyze_file(path: Path):
    import librosa
    import numpy as np
    # ~90s middle excerpt; fall back to whole/short tracks.
    try:
        dur = librosa.get_duration(path=str(path))
    except Exception:
        dur = 0.0
    offset = 30.0 if dur > 150 else 0.0
    want = 90.0 if dur > 30 else None
    y, sr = librosa.load(str(path), sr=22050, mono=True, offset=offset, duration=want)
    if y.size < sr:  # under ~1s of audio -> useless
        raise ValueError("too short / unreadable")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    cam, kname = detect_key(chroma.mean(axis=1))
    energy = float(np.sqrt(np.mean(y ** 2)))
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    dance = pulse_clarity(onset_env, sr)
    lufs = integrated_lufs(y, sr)
    return {"bpm": round(bpm, 1), "key_camelot": cam, "key_name": kname,
            "energy": round(energy, 5), "centroid": round(centroid, 1),
            "mfcc": json.dumps([round(float(x), 3) for x in mfcc]),
            "danceability": round(dance, 4), "lufs": lufs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="music library root (else CRATE_LIB_ROOT / config)")
    ap.add_argument("--buckets", default=None,
                    help="only these top-level folders (CSV), e.g. music,dj,music-mp3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    root = resolve_root(args.root)

    buckets = parse_buckets(args.buckets)
    con = connect(crate_dir(root) / "features.sqlite")
    have = {r[0]: r[1] for r in con.execute("SELECT relpath, mtime FROM features")}
    done = ok = skip = err = 0
    t0 = time.time()
    for p in iter_audio(root, buckets):
        rel = p.relative_to(root).as_posix()
        mt = p.stat().st_mtime
        if not args.rebuild and rel in have and abs(have[rel] - mt) < 1.0:
            skip += 1
            continue
        try:
            f = analyze_file(p)
            con.execute(
                """INSERT INTO features
                   (relpath,mtime,bpm,key_camelot,key_name,energy,centroid,mfcc,
                    danceability,lufs,analyzed_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(relpath) DO UPDATE SET
                     mtime=excluded.mtime, bpm=excluded.bpm, key_camelot=excluded.key_camelot,
                     key_name=excluded.key_name, energy=excluded.energy,
                     centroid=excluded.centroid, mfcc=excluded.mfcc,
                     danceability=excluded.danceability, lufs=excluded.lufs,
                     analyzed_at=excluded.analyzed_at""",
                (rel, mt, f["bpm"], f["key_camelot"], f["key_name"], f["energy"],
                 f["centroid"], f["mfcc"], f["danceability"], f["lufs"], time.time()),
            )
            con.commit()
            ok += 1
            lufs_s = f"{f['lufs']:>6.1f}LUFS" if f["lufs"] is not None else "   --LUFS"
            print(f"OK  {f['bpm']:>5.1f}bpm {f['key_camelot']:>3} dance={f['danceability']:.2f} "
                  f"{lufs_s} {rel}", flush=True)
        except Exception as e:
            err += 1
            print(f"ERR {rel}: {type(e).__name__}: {e}", flush=True)
        done += 1
        if args.limit and done >= args.limit:
            break
    total = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    con.close()
    print(f"\n=== analyzed {ok} new, {skip} skipped, {err} errors in "
          f"{time.time() - t0:.0f}s; features.sqlite now holds {total} ===")


if __name__ == "__main__":
    sys.exit(main())
