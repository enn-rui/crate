#!/usr/bin/env python3
"""analyze_hi.py — HIGH-LEVEL audio features for Crate (genre / mood / danceability / downbeats).

This is the OPTIONAL, heavy tier on top of analyze.py. It feeds smart crates, the journey/set
builder, and better cue placement with semantics librosa can't give cheaply:

  • genre            — top genre label + confidence (Essentia Discogs-EffNet, 400-genre head)
  • danceability     — Essentia's DFA-based Danceability algorithm (no model needed), 0..~3 -> 0..1
  • valence/arousal  — mood/affect (Essentia MTG-Jamendo / DEAM mood models)        [model file]
  • instrumentalness — vocal vs instrumental (Essentia voice/instrumental model)    [model file]
  • first_downbeat / downbeat_bpm / beats — phase-aligned grid (madmom RNN downbeats) [optional]
  • structure        — coarse intro/build/drop/outro segments (novelty over spectral features)

Writes one sidecar: <root>/.crate/hi_features.sqlite (relpath PRIMARY KEY). Idempotent by
relpath+mtime. Every feature is computed INDEPENDENTLY and failures are swallowed per-feature, so
partial deps still produce partial rows (e.g. Essentia present but madmom missing -> genre/mood
filled, downbeats NULL).

────────────────────────────────────────────────────────────────────────────────────────────────
HEAVY + VERSION-FRAGILE — install in a SEPARATE venv from analyze.py/embed_muq (numpy conflict):

  Essentia (genre/mood/danceability/instrumentalness):
    pip install essentia-tensorflow        # needs numpy<2  ->  pip install "numpy<2"
    # pretrained model files (.pb) + their .json metadata go in --models-dir, from:
    #   https://essentia.upf.edu/models/   (e.g. discogs-effnet-bs64-1.pb,
    #   genre_discogs400-discogs-effnet-1.pb, mtg_jamendo_moodtheme-discogs-effnet-1.pb,
    #   voice_instrumental-discogs-effnet-1.pb)

  madmom (downbeats / phase-aligned grid):
    pip install cython mido
    pip install git+https://github.com/CPJKU/madmom  # needs numpy<2 too; bundled RNN weights

Run a capability check first — it tells you exactly what's installed/missing WITHOUT touching audio:
    python analyze_hi.py --selftest [--models-dir DIR]
Then:
    python analyze_hi.py --root <lib_root> [--models-dir DIR] [--limit N] [--rebuild] [--no-madmom]

STATUS: scaffold authored 2026-06-28; NOT yet run end-to-end (Essentia/madmom not installed on the
PC or the box's numpy-2 analyze venv). Verify on the friend's 5090 / a dedicated numpy<2 venv as
part of the box re-analysis batch. Treat numbers as unverified until that run.
────────────────────────────────────────────────────────────────────────────────────────────────
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

from _common import resolve_root, crate_dir, iter_audio

# Essentia model files expected under --models-dir (download from essentia.upf.edu/models/). Each
# classifier head also reads the matching <name>.json for its label list. Missing files -> that
# feature is skipped (logged once), not a hard error.
EMBED_MODEL = "discogs-effnet-bs64-1.pb"                       # shared embedding backbone
GENRE_MODEL = "genre_discogs400-discogs-effnet-1.pb"          # 400-way genre head
MOOD_MODEL = "mtg_jamendo_moodtheme-discogs-effnet-1.pb"      # multi-label mood/theme head
VOICE_MODEL = "voice_instrumental-discogs-effnet-1.pb"        # [instrumental, voice]
SR = 16000  # Essentia TF models are trained at 16 kHz mono


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute(
        """CREATE TABLE IF NOT EXISTS hi_features (
            relpath          TEXT PRIMARY KEY,
            mtime            REAL,
            genre            TEXT,      -- top genre label
            genre_conf       REAL,      -- 0..1 confidence
            danceability     REAL,      -- 0..1 (Essentia Danceability / 3, clamped)
            valence          REAL,      -- 0..1 (mood positivity, if a mood model is present)
            arousal          REAL,      -- 0..1 (mood energy/intensity)
            instrumentalness REAL,      -- 0..1 (1 = instrumental, 0 = vocal-heavy)
            first_downbeat   REAL,      -- seconds to the first downbeat (madmom)
            downbeat_bpm     REAL,      -- tempo implied by the downbeat grid
            beats            TEXT,      -- JSON [ [time_s, beat_in_bar], ... ] (may be truncated)
            structure        TEXT,      -- JSON [ {start,end,label}, ... ]
            analyzed_at      REAL
        )"""
    )
    con.commit()
    return con


# ── Essentia ───────────────────────────────────────────────────────────────────────────────────
class EssentiaFeatures:
    """Lazily loads Essentia + the model files once. Each method returns None if its model/dep is
    unavailable, so the caller can persist whatever did compute."""

    def __init__(self, models_dir: Path | None):
        self.ok = False
        self.models_dir = Path(models_dir) if models_dir else None
        self._embed = self._genre = self._mood = self._voice = None
        self._genre_labels = self._mood_labels = None
        try:
            import essentia.standard as es  # noqa: F401
            self._es = es
            self.ok = True
        except Exception as e:
            self._es = None
            self._err = e

    def _model_path(self, name: str) -> Path | None:
        if not self.models_dir:
            return None
        p = self.models_dir / name
        return p if p.exists() else None

    def _labels(self, name: str):
        meta = self._model_path(name.replace(".pb", ".json"))
        if not meta:
            return None
        try:
            return json.loads(meta.read_text()).get("classes")
        except Exception:
            return None

    def embedding(self, audio):
        """Discogs-EffNet embedding (the shared input for the classifier heads)."""
        if not self.ok:
            return None
        mp = self._model_path(EMBED_MODEL)
        if not mp:
            return None
        if self._embed is None:
            self._embed = self._es.TensorflowPredictEffnetDiscogs(
                graphFilename=str(mp), output="PartitionedCall:1")
        return self._embed(audio)

    def danceability(self, audio):
        """Essentia's standalone Danceability (Detrended Fluctuation Analysis) — needs NO model
        file. Raw range ~0..3; map to 0..1."""
        if not self.ok:
            return None
        try:
            d, _ = self._es.Danceability()(audio)
            return max(0.0, min(1.0, float(d) / 3.0))
        except Exception:
            return None

    def genre(self, emb):
        if emb is None:
            return None, None
        mp = self._model_path(GENRE_MODEL)
        if not mp:
            return None, None
        try:
            if self._genre is None:
                self._genre = self._es.TensorflowPredict2D(
                    graphFilename=str(mp), input="serving_default_model_Placeholder",
                    output="PartitionedCall:0")
                self._genre_labels = self._labels(GENRE_MODEL)
            import numpy as np
            preds = np.asarray(self._genre(emb)).mean(axis=0)  # avg over time patches
            i = int(preds.argmax())
            label = self._genre_labels[i] if self._genre_labels else str(i)
            return label, round(float(preds[i]), 4)
        except Exception:
            return None, None

    def instrumentalness(self, emb):
        if emb is None:
            return None
        mp = self._model_path(VOICE_MODEL)
        if not mp:
            return None
        try:
            if self._voice is None:
                self._voice = self._es.TensorflowPredict2D(
                    graphFilename=str(mp), output="model/Softmax")
            import numpy as np
            preds = np.asarray(self._voice(emb)).mean(axis=0)  # [instrumental, voice]
            return round(float(preds[0]), 4)
        except Exception:
            return None

    def valence_arousal(self, emb):
        """Approx valence/arousal from the MTG-Jamendo mood head (positive moods -> valence,
        intense moods -> arousal). Heuristic, not a calibrated VA regressor."""
        if emb is None:
            return None, None
        mp = self._model_path(MOOD_MODEL)
        if not mp:
            return None, None
        try:
            if self._mood is None:
                self._mood = self._es.TensorflowPredict2D(
                    graphFilename=str(mp), input="serving_default_model_Placeholder",
                    output="PartitionedCall:0")
                self._mood_labels = self._labels(MOOD_MODEL)
            import numpy as np
            preds = np.asarray(self._mood(emb)).mean(axis=0)
            labels = self._mood_labels or []
            pos = {"happy", "fun", "uplifting", "positive", "love", "hopeful", "party"}
            neg = {"sad", "dark", "melancholic", "angry"}
            hi = {"energetic", "fast", "powerful", "party", "aggressive", "heavy"}
            lo = {"calm", "relaxing", "soft", "ambient", "mellow", "slow"}

            def score(keys):
                vals = [preds[i] for i, l in enumerate(labels) if l.lower() in keys]
                return float(np.mean(vals)) if vals else 0.0
            valence = 0.5 + 0.5 * (score(pos) - score(neg))
            arousal = 0.5 + 0.5 * (score(hi) - score(lo))
            return round(max(0.0, min(1.0, valence)), 4), round(max(0.0, min(1.0, arousal)), 4)
        except Exception:
            return None, None


# ── madmom downbeats ─────────────────────────────────────────────────────────────────────────
def downbeats(path: Path, beats_per_bar=(3, 4)):
    """(first_downbeat_s, downbeat_bpm, beats_json) via madmom's RNN downbeat tracker, or
    (None, None, None) if madmom is unavailable / fails. beats_json = [[time, beat_in_bar], ...]."""
    try:
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
        import numpy as np
        act = RNNDownBeatProcessor()(str(path))
        proc = DBNDownBeatTrackingProcessor(beats_per_bar=list(beats_per_bar), fps=100)
        bt = proc(act)  # array of [time, beat_position_in_bar]
        if bt is None or len(bt) == 0:
            return None, None, None
        downs = bt[bt[:, 1] == 1][:, 0]
        first = float(downs[0]) if len(downs) else float(bt[0, 0])
        bpm = None
        if len(bt) > 1:
            d = np.diff(bt[:, 0])
            d = d[(d > 0.2) & (d < 2.0)]  # plausible beat spacing
            if len(d):
                bpm = round(60.0 / float(np.median(d)), 1)
        beats = json.dumps([[round(float(t), 3), int(b)] for t, b in bt[:512]])
        return round(first, 3), bpm, beats
    except Exception:
        return None, None, None


# ── structure (dep-free) ─────────────────────────────────────────────────────────────────────
def structure(path: Path):
    """Coarse section boundaries via a self-similarity novelty curve over MFCC+chroma (librosa
    only). Labels are positional (intro/body…/outro), not semantic. JSON [{start,end,label}]."""
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        if y.size < sr * 5:
            return None
        bounds = librosa.segment.agglomerative(
            np.vstack([librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13),
                       librosa.feature.chroma_cqt(y=y, sr=sr)]), 6)
        times = librosa.frames_to_time(bounds, sr=sr).tolist()
        times = [0.0] + [t for t in times if t > 0] + [float(len(y) / sr)]
        times = sorted(set(round(t, 2) for t in times))
        segs = []
        for i in range(len(times) - 1):
            label = "intro" if i == 0 else ("outro" if i == len(times) - 2 else f"section{i}")
            segs.append({"start": times[i], "end": times[i + 1], "label": label})
        return json.dumps(segs)
    except Exception:
        return None


def selftest(models_dir):
    print("=== analyze_hi selftest ===")
    ess = EssentiaFeatures(models_dir)
    print(f"essentia import : {'OK' if ess.ok else 'MISSING (' + str(getattr(ess, '_err', '')) + ')'}")
    if models_dir:
        for m in (EMBED_MODEL, GENRE_MODEL, MOOD_MODEL, VOICE_MODEL):
            print(f"  model {m:48} {'found' if (Path(models_dir)/m).exists() else 'MISSING'}")
    else:
        print("  (no --models-dir: genre/mood/instrumentalness skipped; Danceability still works)")
    try:
        import madmom  # noqa: F401
        print("madmom import   : OK")
    except Exception as e:
        print(f"madmom import   : MISSING ({type(e).__name__})")
    try:
        import librosa  # noqa: F401
        print("librosa import  : OK (structure available)")
    except Exception:
        print("librosa import  : MISSING")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--models-dir", default=None, help="folder with Essentia .pb + .json models")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-madmom", action="store_true", help="skip downbeat tracking")
    ap.add_argument("--selftest", action="store_true", help="report installed deps + models, then exit")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest(args.models_dir)
        return 0

    root = resolve_root(args.root)
    ess = EssentiaFeatures(args.models_dir)
    con = connect(crate_dir(root) / "hi_features.sqlite")
    have = {r[0]: r[1] for r in con.execute("SELECT relpath, mtime FROM hi_features")}
    ok = skip = err = 0
    t0 = time.time()
    for p in iter_audio(root):
        rel = p.relative_to(root).as_posix()
        mt = p.stat().st_mtime
        if not args.rebuild and rel in have and abs(have[rel] - mt) < 1.0:
            skip += 1
            continue
        try:
            genre = gconf = dance = val = aro = instr = None
            if ess.ok:
                import essentia.standard as es
                audio = es.MonoLoader(filename=str(p), sampleRate=SR)()
                dance = ess.danceability(audio)
                emb = ess.embedding(audio)
                genre, gconf = ess.genre(emb)
                instr = ess.instrumentalness(emb)
                val, aro = ess.valence_arousal(emb)
            first_db = db_bpm = beats = None
            if not args.no_madmom:
                first_db, db_bpm, beats = downbeats(p)
            struct = structure(p)
            con.execute(
                """INSERT INTO hi_features
                   (relpath,mtime,genre,genre_conf,danceability,valence,arousal,instrumentalness,
                    first_downbeat,downbeat_bpm,beats,structure,analyzed_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(relpath) DO UPDATE SET
                     mtime=excluded.mtime, genre=excluded.genre, genre_conf=excluded.genre_conf,
                     danceability=excluded.danceability, valence=excluded.valence,
                     arousal=excluded.arousal, instrumentalness=excluded.instrumentalness,
                     first_downbeat=excluded.first_downbeat, downbeat_bpm=excluded.downbeat_bpm,
                     beats=excluded.beats, structure=excluded.structure,
                     analyzed_at=excluded.analyzed_at""",
                (rel, mt, genre, gconf, dance, val, aro, instr,
                 first_db, db_bpm, beats, struct, time.time()))
            con.commit()
            ok += 1
            print(f"OK  genre={genre or '-':<18} dance={dance if dance is not None else '-'} "
                  f"db={first_db if first_db is not None else '-'} {rel}", flush=True)
        except Exception as e:
            err += 1
            print(f"ERR {rel}: {type(e).__name__}: {e}", flush=True)
        if args.limit and (ok + err) >= args.limit:
            break
    total = con.execute("SELECT COUNT(*) FROM hi_features").fetchone()[0]
    con.close()
    print(f"\n=== hi-features: {ok} new, {skip} skipped, {err} errors in {time.time()-t0:.0f}s; "
          f"hi_features.sqlite holds {total} ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
