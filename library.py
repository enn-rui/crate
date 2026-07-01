r"""Crate library core — index a local music library, search it, and export a
rekordbox-ready playlist + local copies.

Pure logic, no Qt: the GUI calls these functions; everything here is unit-testable and can
also be driven from the CLI (`python library.py index|search|export ...`).

The library root is configurable (in-app ⚙ FOLDERS / first-run / CRATE_LIB_ROOT env); it may
be a local folder or a network mount. Scan roots ("buckets") are user-defined folders under it.
Tracks filed as <Artist>\<Title>.<ext> fall back to the path for artist/title when tags are
missing. Audio analysis (BPM/key/energy + MuQ-MuLan embeddings + map + waveforms) is computed separately by the
`analysis/` pipeline, which writes SQLite sidecars to <lib_root>/.crate/ that this module reads.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# --- where things live (source build vs frozen bundle) ----------------------
def resource_dir() -> Path:
    """Read-only resources shipped WITH the app (skins/, assets/, analysis/). In a PyInstaller
    bundle these are unpacked under _MEIPASS; in the source build they sit next to this file."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


def app_data_dir() -> Path:
    """Writable per-user dir for the index db + config + seeds. Frozen builds must NOT write next to
    the exe (it can be read-only / Program Files), so state goes to the OS per-user app-data dir; the
    source build keeps everything beside the code for easy dev inspection."""
    if getattr(sys, "frozen", False):
        if os.name == "nt":
            base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        d = base / "Crate"
    else:
        d = Path(__file__).resolve().parent
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- config -----------------------------------------------------------------
def _resolve_lib_root() -> Path:
    """The library root all sidecars (.crate/*.sqlite) + relpaths key off. Resolution order:
    lib_root in crate_config.json (set by ⚙ FOLDERS / first-run) > CRATE_LIB_ROOT env > Z:\\ default.
    The local analysis scripts (analysis/_common.py) resolve the SAME value, so they agree."""
    try:
        cfgp = app_data_dir() / "crate_config.json"
        if cfgp.exists():
            v = json.loads(cfgp.read_text(encoding="utf-8")).get("lib_root")
            if v:
                return Path(v)
    except Exception:
        pass
    return Path(os.environ.get("CRATE_LIB_ROOT", r"Z:\\"))


LIB_ROOT = _resolve_lib_root()
# (label, path-relative-to-LIB_ROOT). The label becomes each track's `bucket`; the path is
# where that bucket lives on disk. The lossless library is split per-track by danceability into
# music/dj (danceable) + music/personal (not), with lossy files in their own music-mp3 section.
BUCKETS = (("dj", "music/dj"), ("personal", "music/personal"), ("mp3", "music-mp3"))
AUDIO_EXTS = {".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wav", ".aiff", ".aif"}
DB_PATH = app_data_dir() / "crate.db"
# features.sqlite is written by analyze.py into <lib_root>/.crate/
FEATURES_PATH = LIB_ROOT / ".crate" / "features.sqlite"
# deleted tracks are MOVED here (reversible) rather than hard-deleted, so a delete is always undoable
QUARANTINE = LIB_ROOT / ".crate" / "trash"
# UMAP 2D coords from the music embeddings (written by analysis: embed_muq.py -> umap_music.py)
UMAP_PATH = LIB_ROOT / ".crate" / "umap.sqlite"
# ARTIST-level UMAP (umap_artists.py): each artist = mean of their tracks' vectors -> 2D
ARTIST_UMAP_PATH = LIB_ROOT / ".crate" / "artist_umap.sqlite"
# full 512-d sonic vectors (MuQ-MuLan, embed_muq.py). The map above is display-only; SIMILARITY/MIXABILITY
# math runs on these full vectors (cosine), which is why the PC reads them directly.
VECTORS_PATH = LIB_ROOT / ".crate" / "music_vectors.sqlite"
# full-d HDBSCAN cluster labels from the box analysis; -1 means noise / unclustered
CLUSTERS_PATH = LIB_ROOT / ".crate" / "clusters.sqlite"
# colored 3-band waveforms (written by the box: waveform.py); PC reads + renders, soundfile fallback
WAVEFORM_PATH = LIB_ROOT / ".crate" / "waveforms.sqlite"
DEFAULT_EXPORT_ROOT = Path(
    os.environ.get("CRATE_EXPORT_ROOT", str(Path.home() / "Music" / "DJ" / "incoming"))
)
# Saved crates live as folders under here (file copies + .m3u8 + a sources manifest). The folder
# IS the persistent crate: the app lists + reopens these. Configurable via crate_config.json.
DEFAULT_CRATES_ROOT = Path(
    os.environ.get("CRATE_CRATES_ROOT", str(Path.home() / "Music" / "DJ" / "crates"))
)
CONFIG_PATH = app_data_dir() / "crate_config.json"
CRATE_MANIFEST = ".crate-sources.txt"   # one original library path per line, inside a crate folder


# --- config (scan roots + crates root, user-editable in the app) ------------
def default_config() -> dict:
    """The out-of-box config: scan the three Z: buckets, save crates to ~/Music/DJ/crates."""
    return {
        "scan_roots": [{"label": lbl, "path": str(LIB_ROOT / rel)} for lbl, rel in BUCKETS],
        "crates_root": str(DEFAULT_CRATES_ROOT),
        "skin": "terminal",
    }


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = default_config()
    try:
        if Path(path).exists():
            saved = json.loads(Path(path).read_text(encoding="utf-8"))
            if saved.get("scan_roots"):
                cfg["scan_roots"] = saved["scan_roots"]
            if saved.get("crates_root"):
                cfg["crates_root"] = saved["crates_root"]
            if saved.get("skin"):
                cfg["skin"] = saved["skin"]
            if saved.get("lib_root"):
                cfg["lib_root"] = saved["lib_root"]
            if saved.get("analysis_python"):
                cfg["analysis_python"] = saved["analysis_python"]
            if saved.get("analysis_remote"):
                cfg["analysis_remote"] = saved["analysis_remote"]
    except Exception:
        pass
    return cfg


def save_config(cfg: dict, path: Path = CONFIG_PATH) -> None:
    Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def set_lib_root(folder: str) -> None:
    """Point Crate at a music library root: persist lib_root to config AND update the live module
    paths (LIB_ROOT + the .crate sidecar paths) so the running session uses it without a restart.
    Used by first-run setup; the analysis scripts read the same lib_root from config."""
    global LIB_ROOT, FEATURES_PATH, QUARANTINE, UMAP_PATH, ARTIST_UMAP_PATH, WAVEFORM_PATH, VECTORS_PATH, CLUSTERS_PATH
    LIB_ROOT = Path(folder)
    FEATURES_PATH = LIB_ROOT / ".crate" / "features.sqlite"
    QUARANTINE = LIB_ROOT / ".crate" / "trash"
    UMAP_PATH = LIB_ROOT / ".crate" / "umap.sqlite"
    ARTIST_UMAP_PATH = LIB_ROOT / ".crate" / "artist_umap.sqlite"
    WAVEFORM_PATH = LIB_ROOT / ".crate" / "waveforms.sqlite"
    VECTORS_PATH = LIB_ROOT / ".crate" / "music_vectors.sqlite"
    CLUSTERS_PATH = LIB_ROOT / ".crate" / "clusters.sqlite"
    clear_vector_cache()  # vectors are keyed to the old root; drop the cache on a root change
    cfg = load_config()
    cfg["lib_root"] = str(folder)
    save_config(cfg)


def get_crates_root() -> Path:
    return Path(load_config()["crates_root"])


def get_skin() -> str:
    return load_config().get("skin", "terminal")


def set_skin(key: str) -> None:
    cfg = load_config()
    cfg["skin"] = key
    save_config(cfg)


def config_sources() -> list[tuple[str, Path]]:
    """[(label, abspath)] scan sources from config — each label becomes the track 'bucket'."""
    return [(r["label"], Path(r["path"])) for r in load_config()["scan_roots"]]


@dataclass
class Track:
    path: str
    bucket: str
    artist: str
    title: str
    album: str
    ext: str
    size: int
    mtime: float
    duration: float  # seconds, 0 if unknown
    bpm: float | None = None
    key: str | None = None
    rating: int | None = None
    energy: float | None = None
    color: str | None = None
    comment: str | None = None
    danceability: float | None = None   # 0..1 pulse-clarity (analyze.py)
    lufs: float | None = None           # integrated loudness, dB (analyze.py)
    # populated by the GUI (TrackModel.set_tracks) for the table's TAGS / CUE columns — not stored
    tag_summary: str = ""
    cue_count: int = 0


# --- db ---------------------------------------------------------------------
_SCHEMA_READY: set[str] = set()   # db paths whose schema has been ensured this process


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a connection, ensuring the schema exists. The DDL (CREATE/ALTER/INDEX) is run only
    ONCE per db path per process — the auto-saving tag/cue UI opens many short-lived connections,
    and re-running all the migrations on every one was pure overhead."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    # Per-connection pragmas (busy_timeout + synchronous are NOT persisted in the db file, unlike
    # journal_mode, so they must be set on every connection): the auto-saving tag/cue UI writes many
    # small transactions while a background worker may be indexing; WAL lets a read proceed during a
    # write, busy_timeout avoids an instant "database is locked" on concurrent access, and NORMAL sync
    # is safe under WAL and far cheaper than the default FULL fsync per commit.
    try:
        con.execute("PRAGMA busy_timeout=5000")
        con.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    if str(db_path) not in _SCHEMA_READY:
        try:
            con.execute("PRAGMA journal_mode=WAL")   # persisted in the db file; set once is enough
        except Exception:
            pass
        _ensure_schema(con)
        _SCHEMA_READY.add(str(db_path))
    return con


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            path     TEXT PRIMARY KEY,
            bucket   TEXT,
            artist   TEXT,
            title    TEXT,
            album    TEXT,
            ext      TEXT,
            size     INTEGER,
            mtime    REAL,
            duration REAL,
            bpm      REAL,
            key      TEXT
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tracks_title  ON tracks(title)")
    # search() / list_buckets() / harmonic_matches() filter or DISTINCT on these — index them so the
    # queries don't full-scan the tracks table as the library grows.
    con.execute("CREATE INDEX IF NOT EXISTS idx_tracks_bucket ON tracks(bucket)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tracks_key    ON tracks(key)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tracks_bpm    ON tracks(bpm)")
    # migrations: feature columns added by the box analysis (Phase 2)
    cols = {r[1] for r in con.execute("PRAGMA table_info(tracks)")}
    for col, decl in (("energy", "REAL"), ("centroid", "REAL"), ("mfcc", "TEXT"),
                      ("rating", "INTEGER"), ("color", "TEXT"), ("comment", "TEXT"),
                      ("lufs", "REAL"), ("danceability", "REAL")):
        if col not in cols:
            con.execute(f"ALTER TABLE tracks ADD COLUMN {col} {decl}")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS track_tags (
            path     TEXT,
            category TEXT,
            value    TEXT,
            PRIMARY KEY(path, category, value)
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_track_tags_category ON track_tags(category)")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS track_cues (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            path        TEXT,
            kind        TEXT,
            idx         TEXT,
            position_ms INTEGER,
            color       TEXT,
            name        TEXT
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_track_cues_path ON track_cues(path)")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS smart_crates (
            name    TEXT PRIMARY KEY,
            spec    TEXT,              -- JSON {match: all|any, conditions: [...]}
            created REAL
        )
        """
    )
    # Virtual per-artist bucket override. The physical folders (music/dj, music/personal,
    # music-mp3) were split per-track by librosa danceability, which mislabels genre (steady-beat
    # hip-hop/R&B scored "danceable") AND scatters one artist across both. This table makes the
    # dj/personal bucket a VIRTUAL, per-artist property the user curates — no files move. It wins
    # over the folder-derived bucket and is re-applied after every index. Key = lowercased primary
    # artist; bucket = 'dj' | 'personal' (or any label).
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS artist_buckets (
            artist TEXT PRIMARY KEY,   -- lowercased primary artist (see primary_artist())
            bucket TEXT
        )
        """
    )
    # Quarantine manifest: maps a trashed file's path-inside-the-trash back to its EXACT original
    # absolute location, so restore returns it where it came from even when it was deleted from an
    # added scan root OUTSIDE lib_root (those only kept a basename before, so restore misplaced them).
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS quarantine (
            trash_rel TEXT PRIMARY KEY,   -- posix relpath under the trash root
            orig_path TEXT,               -- original absolute path the file was deleted from
            deleted   REAL
        )
        """
    )
    con.commit()


# --- tag reading ------------------------------------------------------------
def _first(tagval) -> str:
    if tagval is None:
        return ""
    if isinstance(tagval, (list, tuple)):
        return str(tagval[0]) if tagval else ""
    return str(tagval)


def read_tags(path: Path) -> tuple[str, str, str, float, float | None]:
    """Return (artist, title, album, duration, bpm). Falls back to <Artist>\\<Title> path."""
    artist = title = album = ""
    duration = 0.0
    bpm: float | None = None
    try:
        import mutagen

        mf = mutagen.File(str(path), easy=True)
        if mf is not None:
            artist = _first(mf.get("artist"))
            title = _first(mf.get("title"))
            album = _first(mf.get("album"))
            bpm_raw = _first(mf.get("bpm"))
            if bpm_raw:
                try:
                    bpm = float(bpm_raw)
                except ValueError:
                    bpm = None
            if getattr(mf, "info", None) is not None:
                duration = float(getattr(mf.info, "length", 0.0) or 0.0)
    except Exception:
        pass
    # path fallbacks: file lives at <Artist>\<Title>.<ext>
    if not title:
        title = path.stem
    if not artist:
        artist = path.parent.name
    return artist.strip(), title.strip(), album.strip(), duration, bpm


# --- indexing ---------------------------------------------------------------
def iter_audio_sources(sources):
    """sources = [(label, abspath)]; yields (label, file) for every audio file under each."""
    for label, root in sources:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                yield label, p


def index(root: Path = LIB_ROOT, db_path: Path = DB_PATH, buckets=BUCKETS,
          progress=None, sources=None) -> dict:
    """Walk the library, upsert changed/new tracks, prune deleted. Returns counts.

    `sources` = [(label, abspath)] overrides root/buckets (used by the GUI via config_sources());
    when omitted, scans <root>/<bucket> for each bucket (back-compat default / tests).
    `progress` is an optional callable(done:int, label:str) for the GUI.
    Skips files whose path+size+mtime already match the DB row (fast re-index).
    """
    if sources is None:
        sources = [(lbl, Path(root) / rel) for lbl, rel in buckets]
    _maybe_seed_buckets(db_path)   # restore curated buckets on a fresh db before applying them
    con = connect(db_path)
    existing = {
        row["path"]: (row["size"], row["mtime"], row["bucket"])
        for row in con.execute("SELECT path, size, mtime, bucket FROM tracks")
    }
    overrides = get_artist_buckets(db_path)   # virtual per-artist buckets win over the folder label

    def eff_bucket(path: str, folder_bucket: str, artist: str = "") -> str:
        return (overrides.get(_folder_artist(path))
                or (overrides.get(primary_artist(artist)) if artist else None)
                or folder_bucket)

    seen: set[str] = set()
    added = updated = skipped = 0
    done = 0
    for bucket, p in iter_audio_sources(sources):
        key = str(p)
        seen.add(key)
        st = p.stat()
        prev = existing.get(key)
        if prev is not None and prev[0] == st.st_size and abs(prev[1] - st.st_mtime) < 1.0:
            # unchanged file: skip the tag re-read, but self-heal a stale bucket label cheaply —
            # toward the virtual override if the artist has one, else the folder label (e.g. a
            # scan-root was relabeled). This keeps the skip-path consistent with apply_artist_buckets.
            want = eff_bucket(key, bucket)
            if prev[2] != want:
                con.execute("UPDATE tracks SET bucket=? WHERE path=?", (want, key))
            skipped += 1
        else:
            artist, title, album, duration, bpm = read_tags(p)
            con.execute(
                """INSERT INTO tracks(path,bucket,artist,title,album,ext,size,mtime,duration,bpm,key)
                   VALUES(?,?,?,?,?,?,?,?,?,?,COALESCE((SELECT key FROM tracks WHERE path=?),NULL))
                   ON CONFLICT(path) DO UPDATE SET
                     bucket=excluded.bucket, artist=excluded.artist, title=excluded.title,
                     album=excluded.album, ext=excluded.ext, size=excluded.size,
                     mtime=excluded.mtime, duration=excluded.duration,
                     bpm=COALESCE(excluded.bpm, tracks.bpm)""",
                (key, eff_bucket(key, bucket, artist), artist, title, album, p.suffix.lower(),
                 st.st_size, st.st_mtime, duration, bpm, key),
            )
            if prev is None:
                added += 1
            else:
                updated += 1
        done += 1
        if progress and done % 50 == 0:
            progress(done, f"{bucket}/{p.name}")
    # prune deleted — but ONLY when it's safe. If a scan root is unreachable (e.g. Z: is
    # disconnected) or the whole walk found nothing while the DB has tracks, pruning would
    # wipe the index AND the ratings (which live only here). Skip pruning in those cases.
    missing_roots = [str(p) for _label, p in sources if not Path(p).exists()]
    do_prune = (not missing_roots) and bool(seen or not existing)
    removed = 0
    if do_prune:
        for key in list(existing):
            if key not in seen:
                con.execute("DELETE FROM tracks WHERE path=?", (key,))
                removed += 1
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    con.close()
    matched = sync_features(db_path=db_path)  # pull in any analysis the box has done
    rebucketed = apply_artist_buckets(db_path)  # re-apply virtual dj/personal over the folder split
    return {"added": added, "updated": updated, "skipped": skipped,
            "removed": removed, "total": total, "features": matched,
            "rebucketed": rebucketed, "pruned": do_prune, "missing_roots": missing_roots}


def index_from_config(db_path: Path = DB_PATH, progress=None) -> dict:
    """GUI entry point: index every scan root configured in crate_config.json."""
    return index(db_path=db_path, sources=config_sources(), progress=progress)


# --- virtual per-artist buckets (dj/personal as a curated tag, not a folder) -------------------
_PRIMARY_RE = re.compile(r"\s*(?:;|,|/|&|\bfeat\.?\b|\bft\.?\b|\bx\b|\bvs\.?\b)\s*", re.I)


def primary_artist(name: str) -> str:
    """First-billed artist, lowercased — the override key. 'Charli XCX feat. SOPHIE' -> 'charli xcx'.
    Mirrors analysis/umap_artists.primary_artist so map + buckets agree on what 'an artist' is."""
    if not name:
        return ""
    return _PRIMARY_RE.split(name, maxsplit=1)[0].strip().lower()


def _folder_artist(path: str) -> str:
    """Lowercased parent-folder name of a track path (= the filing artist), the override fallback
    key for when the embedded tag differs from how the file is filed (<bucket>/<Artist>/<file>)."""
    parts = Path(path).as_posix().split("/")
    return parts[-2].lower() if len(parts) >= 2 else ""


def list_buckets(db_path: Path = DB_PATH) -> list[str]:
    """Distinct bucket labels actually present in the index (drives the filter dropdown), ordered
    dj, personal, then any others alphabetically. Reflects the virtual artist buckets, so an emptied
    folder label (e.g. 'mp3' after everything is re-tagged dj/personal) no longer shows."""
    con = connect(db_path)
    try:
        present = {r[0] for r in con.execute(
            "SELECT DISTINCT bucket FROM tracks WHERE bucket IS NOT NULL AND bucket != ''")}
    finally:
        con.close()
    order = [b for b in ("dj", "personal") if b in present]
    return order + sorted(present - set(order))


def get_artist_buckets(db_path: Path = DB_PATH) -> dict[str, str]:
    """{lowercased primary artist: bucket} — the full override map."""
    con = connect(db_path)
    try:
        return {r[0]: r[1] for r in con.execute("SELECT artist, bucket FROM artist_buckets")}
    finally:
        con.close()


def artist_key(path: str, artist: str = "") -> str:
    """The canonical override key for a track: its FILING-FOLDER artist (clean + stable — how the
    UMAP keys artists too), falling back to the tag's primary artist. Embedded tags are messy
    ('UGK (Underground Kingz)', 'Scarface Rapper') and don't reduce to the folder name, so the
    folder wins."""
    return _folder_artist(path) or primary_artist(artist)


def apply_artist_buckets(db_path: Path = DB_PATH) -> int:
    """Re-write tracks.bucket from the artist_buckets overrides (match by filing folder, else by
    tag-primary artist). Idempotent; called at the end of index() so a re-index never reverts the
    user's curation back to the danceability folder split. Returns rows changed."""
    overrides = get_artist_buckets(db_path)
    if not overrides:
        return 0
    con = connect(db_path)
    changed = 0
    try:
        for path, artist, bucket in con.execute("SELECT path, artist, bucket FROM tracks").fetchall():
            want = overrides.get(_folder_artist(path)) or overrides.get(primary_artist(artist))
            if want and want != bucket:
                con.execute("UPDATE tracks SET bucket=? WHERE path=?", (want, path))
                changed += 1
        con.commit()
    finally:
        con.close()
    return changed


def set_artist_bucket(artist: str, bucket: str, db_path: Path = DB_PATH, path: str = "") -> int:
    """Assign an artist (and ALL their tracks) to a bucket — the one-click 'move this artist to
    dj/personal'. Pass a `path` from one of the artist's tracks so the canonical filing-folder key
    is used (preferred); otherwise the tag's primary artist. Persists the override, re-applies to
    every track, returns total rows moved."""
    key = artist_key(path, artist) if path else primary_artist(artist)
    if not key:
        return 0
    con = connect(db_path)
    try:
        con.execute(
            "INSERT INTO artist_buckets(artist, bucket) VALUES(?, ?) "
            "ON CONFLICT(artist) DO UPDATE SET bucket=excluded.bucket", (key, bucket))
        con.commit()
    finally:
        con.close()
    return apply_artist_buckets(db_path)   # Python-side match (handles backslash paths + messy tags)


def _maybe_seed_buckets(db_path: Path = DB_PATH) -> None:
    """If the override table is empty but a recovery `artist_buckets_seed.json` sits next to the db
    (written whenever a seed is installed), load it — so a rebuilt crate.db restores the curated
    dj/personal classification instead of falling back to the danceability folder split. Never
    overrides existing edits (only runs when the table is empty)."""
    seed = Path(db_path).parent / "artist_buckets_seed.json"
    if not seed.exists():
        return
    con = connect(db_path)
    try:
        empty = con.execute("SELECT COUNT(*) FROM artist_buckets").fetchone()[0] == 0
    finally:
        con.close()
    if empty:
        try:
            seed_artist_buckets(json.loads(seed.read_text(encoding="utf-8")), db_path=db_path)
        except Exception:
            pass


def seed_artist_buckets(mapping: dict[str, str], db_path: Path = DB_PATH) -> dict:
    """Bulk-set overrides from {artist-folder-name: bucket} then apply to tracks. Keys are
    lowercased as filing-folder artist names (NOT tag-primary — see artist_key). Used to install an
    initial classification. Returns {artists, tracks_changed}."""
    con = connect(db_path)
    try:
        for art, buck in mapping.items():
            con.execute(
                "INSERT INTO artist_buckets(artist, bucket) VALUES(?, ?) "
                "ON CONFLICT(artist) DO UPDATE SET bucket=excluded.bucket",
                (art.strip().lower(), buck))
        con.commit()
    finally:
        con.close()
    return {"artists": len(mapping), "tracks_changed": apply_artist_buckets(db_path)}


def sync_features(db_path: Path = DB_PATH, features_path: Path = FEATURES_PATH,
                  lib_root: Path = LIB_ROOT) -> int:
    """Merge BPM/key/energy from the box's features.sqlite (read over Z:) into crate.db.

    The sidecar keys rows by relpath under the library root (e.g. 'music/Artist/Title.flac');
    map that onto the local Windows path to match the tracks table. Returns rows updated.
    """
    if not Path(features_path).exists():
        return 0
    con = connect(db_path)
    matched = 0
    try:
        # copy the sidecar locally first — avoids SMB read-locks / partial reads while the
        # box is still writing to it.
        import tempfile
        tmp = Path(tempfile.gettempdir()) / "crate_features_snapshot.sqlite"
        shutil.copy2(features_path, tmp)
        fcon = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        # lufs/danceability are newer analyze.py columns; an older box sidecar may lack them, so
        # only pull them when present (and map NULL -> leave column alone via COALESCE on UPDATE).
        fcols = {r[1] for r in fcon.execute("PRAGMA table_info(features)")}
        has_lufs = "lufs" in fcols
        has_dance = "danceability" in fcols
        extra = (", lufs" if has_lufs else "") + (", danceability" if has_dance else "")
        rows = fcon.execute(
            f"SELECT relpath, bpm, key_camelot, energy, centroid, mfcc{extra} FROM features").fetchall()
        fcon.close()
        for row in rows:
            rel, bpm, cam, energy, centroid, mfcc = row[:6]
            local = str(Path(lib_root) / rel)  # Z:\music\Artist\Title.flac
            i = 6
            lufs = row[i] if has_lufs else None
            i += 1 if has_lufs else 0
            dance = row[i] if has_dance else None
            cur = con.execute(
                "UPDATE tracks SET bpm=?, key=?, energy=?, centroid=?, mfcc=?, "
                "lufs=COALESCE(?, lufs), danceability=COALESCE(?, danceability) WHERE path=?",
                (bpm, cam, energy, centroid, mfcc, lufs, dance, local))
            matched += cur.rowcount
        con.commit()
    except Exception:
        pass
    finally:
        con.close()
    return matched


# --- in-app analysis runner -------------------------------------------------
ANALYSIS_DIR = resource_dir() / "analysis"


def analysis_exe_path() -> Path | None:
    """In the self-contained build the heavy pipeline is frozen into its OWN exe (`crate-analyze`)
    sitting next to the app. Returns it when we're running frozen and it's present, else None (the
    source/dev build runs analysis through a Python venv instead — see analysis_python_path)."""
    if not getattr(sys, "frozen", False):
        return None
    name = "crate-analyze.exe" if os.name == "nt" else "crate-analyze"
    cand = Path(sys.executable).resolve().parent / name
    return cand if cand.exists() else None


def analysis_available() -> bool:
    """True when ANALYZE can run locally — either the bundled exe or a set-up analysis venv exists."""
    return analysis_exe_path() is not None or analysis_python_path() is not None


def analysis_python_path() -> Path | None:
    """Locate the interpreter for the HEAVY analysis venv (torch + librosa + transformers + umap,
    ~2 GB — deliberately kept out of the app's light venv). Resolution order:
      1. config 'analysis_python'  2. analysis/.venv/Scripts/python.exe (Windows)
      3. analysis/.venv/bin/python (POSIX)
    Returns None if no such env has been set up (the main rig runs analysis on the box instead)."""
    p = load_config().get("analysis_python")
    if p and Path(p).exists():
        return Path(p)
    base = ANALYSIS_DIR / ".venv"
    for c in (base / "Scripts" / "python.exe", base / "bin" / "python"):
        if c.exists():
            return c
    return None


def run_analysis(root: Path = None, python_exe: str | None = None, rebuild: bool = False,
                 progress=None) -> dict:
    """Run Crate's full local analysis pipeline (analysis/analyze_all.py) against `root`, in-app —
    BPM/key/energy -> MuQ-MuLan embeddings -> map -> waveforms, all written to <root>/.crate/ where
    the app reads them. Streams step headers via progress(n, label). Returns
    {ok, code, root, log}. Raises RuntimeError if no analysis interpreter is available.

    This shells out to the heavy analysis venv (see analysis_python_path) so the GUI process stays
    light; call it from a worker thread (it blocks for minutes)."""
    import subprocess
    root = Path(root) if root else LIB_ROOT
    exe = analysis_exe_path()
    if exe:                                     # bundled build: drive the frozen analysis exe directly
        cmd = [str(exe), "--root", str(root)]
        cwd = None
    else:                                       # source/dev build: shell the heavy analysis venv
        py = Path(python_exe) if python_exe else analysis_python_path()
        if not py:
            raise RuntimeError(
                "No analysis environment found. Create one (~2 GB):\n"
                "  python -m venv analysis/.venv\n"
                "  analysis/.venv/Scripts/pip install -r analysis/requirements-analysis.txt\n"
                "  (plus the torch build for your GPU — see the requirements file)\n"
                "or set 'analysis_python' in crate_config.json.")
        cmd = [str(py), str(ANALYSIS_DIR / "analyze_all.py"), "--root", str(root)]
        cwd = str(ANALYSIS_DIR)
    if rebuild:
        cmd.append("--rebuild")
    # Read the child's stdout as UTF-8 (errors="replace") and tell the child to emit UTF-8, so a track
    # name with accents/Unicode can't raise UnicodeEncodeError on a Windows cp1252 console and kill the
    # run. (text=True alone would decode with the GUI's locale codec and could choke on the same bytes.)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1, cwd=cwd, env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log: list[str] = []
    n = 0
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        log.append(line)
        if progress and line.startswith("====="):     # a pipeline step header -> a clean progress tick
            n += 1
            progress(n, line.strip("= ").strip())
    proc.wait()
    return {"ok": proc.returncode == 0, "code": proc.returncode, "root": str(root), "log": log[-20:]}


def analysis_remote_config() -> dict | None:
    """Optional remote mode: run the analysis pipeline on another machine over SSH instead of the
    local analysis venv (handy if a NAS / server hosts the library and does the heavy compute; the
    app just triggers it, then reads the sidecars it writes under the mounted library root). Opt in
    per-machine via crate_config.json:
        "analysis_remote": {"ssh": "user@host",
                            "python": "/path/to/analysis-venv/bin/python",
                            "script": "/path/to/crate/analysis/analyze_all.py",
                            "root":   "/path/to/music"}
    Only 'ssh' is required; the rest fall back to the defaults below. Returns the resolved dict, or
    None when not configured (the normal case — the app uses its local/bundled analysis instead)."""
    rc = load_config().get("analysis_remote")
    if not isinstance(rc, dict) or not rc.get("ssh"):
        return None
    return {
        "ssh": rc["ssh"],
        "python": rc.get("python", "~/crate/.venv/bin/python"),
        "script": rc.get("script", "~/crate/analysis/analyze_all.py"),
        "root": rc.get("root", "/path/to/music"),
        "rebuild": bool(rc.get("rebuild", False)),
    }


def run_analysis_remote(rebuild: bool = False, progress=None, cfg: dict | None = None) -> dict:
    """Refresh analysis on the remote box over SSH (personal rig: the box does all compute, the PC
    triggers it then reads the resulting sidecars via LIB_ROOT). Runs the box's analyze_all.py
    (idempotent — only new/changed tracks), streaming its '=====' step headers via progress(n, label).
    Returns {ok, code, root, log}. Call from a worker thread; it blocks for the length of the run."""
    import subprocess, shlex
    rc = cfg or analysis_remote_config()
    if not rc:
        raise RuntimeError("No 'analysis_remote' configured in crate_config.json.")
    # ssh runs this string through the REMOTE shell, so every config-supplied component is quoted to
    # stop metacharacters in a (user-editable) config from injecting shell commands on the far side.
    def _rq(s: str) -> str:
        s = str(s)
        if s.startswith("~/"):                     # keep remote-home (~) expansion, quote the rest
            return "~/" + shlex.quote(s[2:])
        return shlex.quote(s)
    dest = str(rc["ssh"])
    if dest.startswith("-"):                       # don't let the host masquerade as an ssh option
        raise RuntimeError(f"Refusing suspicious SSH destination: {dest!r}")
    remote = f"{_rq(rc['python'])} {_rq(rc['script'])} --root {_rq(rc['root'])}"
    if rebuild or rc.get("rebuild"):
        remote += " --rebuild"
    cmd = ["ssh", dest, remote]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log: list[str] = []
    n = 0
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        log.append(line)
        if progress and line.startswith("====="):     # a pipeline step header -> a clean progress tick
            n += 1
            progress(n, line.strip("= ").strip())
    proc.wait()
    return {"ok": proc.returncode == 0, "code": proc.returncode, "root": rc["root"], "log": log[-20:]}


# --- search -----------------------------------------------------------------
def _row_to_track(r: sqlite3.Row) -> Track:
    keys = r.keys()
    return Track(r["path"], r["bucket"], r["artist"], r["title"], r["album"], r["ext"],
                 r["size"], r["mtime"], r["duration"], r["bpm"], r["key"], r["rating"],
                 r["energy"], r["color"], r["comment"],
                 r["danceability"] if "danceability" in keys else None,
                 r["lufs"] if "lufs" in keys else None)


def search(query: str = "", bucket: str | None = None,
           bpm_range: tuple[float, float] | None = None, key: str | None = None,
           limit: int = 1000, db_path: Path = DB_PATH) -> list[Track]:
    con = connect(db_path)
    sql = "SELECT * FROM tracks WHERE 1=1"
    args: list = []
    if query:
        sql += " AND (artist LIKE ? OR title LIKE ? OR album LIKE ?)"
        like = f"%{query}%"
        args += [like, like, like]
    if bucket:
        sql += " AND bucket=?"
        args.append(bucket)
    if bpm_range:
        sql += " AND bpm IS NOT NULL AND bpm > 0 AND bpm BETWEEN ? AND ?"
        args += [bpm_range[0], bpm_range[1]]
    if key:
        sql += " AND key=?"
        args.append(key)
    sql += " ORDER BY artist COLLATE NOCASE, title COLLATE NOCASE LIMIT ?"
    args.append(limit)
    rows = con.execute(sql, args).fetchall()
    con.close()
    return [_row_to_track(r) for r in rows]


# --- smart crates (dynamic rule-based playlists) ----------------------------
# A smart crate is a saved rule SPEC, not a folder of files: it RESOLVES live against the index
# every time it's opened (rekordbox "Intelligent Playlist" / Lexicon "smart list" model). Spec:
#   {"match": "all"|"any", "conditions": [ {"field":..., "op":..., "value":...}, ... ]}
# Saved/folder crates ([[save_crate]]) stay separate — those are frozen copies for export.

# field -> (sql column or None for special handling, kind)
_SMART_NUM_FIELDS = {"bpm", "rating", "energy", "danceability", "lufs", "duration", "year"}
_SMART_TEXT_FIELDS = {"artist", "title", "album", "comment"}


def _smart_condition_sql(cond: dict) -> tuple[str, list] | None:
    """Translate one condition into (sql_fragment, args), or None to skip a malformed one.
    Everything is parameterized — values never go into the SQL string."""
    field = (cond.get("field") or "").lower()
    op = (cond.get("op") or "").lower()
    val = cond.get("value")
    try:
        if field in _SMART_NUM_FIELDS:
            col = field
            if op == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
                return f"({col} IS NOT NULL AND {col} BETWEEN ? AND ?)", [float(val[0]), float(val[1])]
            if op in (">=", "gte"):
                return f"({col} IS NOT NULL AND {col} >= ?)", [float(val)]
            if op in ("<=", "lte"):
                return f"({col} IS NOT NULL AND {col} <= ?)", [float(val)]
            if op in ("=", "is"):
                return f"({col} = ?)", [float(val)]
            return None
        if field == "key":
            if not val:
                return None
            if op == "harmonic":                      # key + all Camelot-compatible keys
                neigh = list(camelot_neighbors(str(val)).keys()) or [str(val)]
                return f"(key IN ({','.join('?' * len(neigh))}))", neigh
            if op in ("is", "="):
                return "(key = ?)", [str(val)]
            return None
        if field == "bucket":
            return "(bucket = ?)", [str(val)]
        if field in _SMART_TEXT_FIELDS:
            if op in ("is", "="):
                return f"({field} = ?)", [str(val)]
            if op in ("not_contains", "excludes"):
                return f"(COALESCE({field},'') NOT LIKE ?)", [f"%{val}%"]
            return f"({field} LIKE ?)", [f"%{val}%"]   # default: contains
        if field == "text":                            # artist/title/album free-text
            like = f"%{val}%"
            return "(artist LIKE ? OR title LIKE ? OR album LIKE ?)", [like, like, like]
        if field == "tag":                             # value = [category, value] or "category:value"
            cat, tval = (val if isinstance(val, (list, tuple)) and len(val) == 2
                         else (str(val).split(":", 1) + [""])[:2])
            sub = ("SELECT 1 FROM track_tags tt WHERE tt.path = tracks.path "
                   "AND tt.category = ? AND tt.value = ?")
            if op in ("not_has", "excludes"):
                return f"(NOT EXISTS ({sub}))", [str(cat), str(tval)]
            return f"(EXISTS ({sub}))", [str(cat), str(tval)]
        if field == "rated":                           # has any rating / unrated
            return ("(rating IS NOT NULL AND rating > 0)" if op != "no"
                    else "(rating IS NULL OR rating = 0)"), []
        if field == "color":
            if op in ("not", "excludes"):
                return "(COALESCE(color,'') != ?)", [str(val)]
            return "(color = ?)", [str(val)]
    except (TypeError, ValueError):
        return None
    return None


def evaluate_smart_crate(spec: dict, db_path: Path = DB_PATH, limit: int = 5000) -> list[Track]:
    """Resolve a smart-crate spec to the matching tracks, live. match='any' OR-combines the
    conditions, anything else AND-combines them. An empty/invalid spec returns nothing."""
    conds = (spec or {}).get("conditions") or []
    frags, args = [], []
    for c in conds:
        built = _smart_condition_sql(c)
        if built:
            frags.append(built[0])
            args.extend(built[1])
    if not frags:
        return []
    joiner = " OR " if (spec or {}).get("match") == "any" else " AND "
    sql = ("SELECT * FROM tracks WHERE " + joiner.join(frags) +
           " ORDER BY artist COLLATE NOCASE, title COLLATE NOCASE LIMIT ?")
    con = connect(db_path)
    rows = con.execute(sql, [*args, limit]).fetchall()
    con.close()
    return [_row_to_track(r) for r in rows]


def save_smart_crate(name: str, spec: dict, db_path: Path = DB_PATH) -> None:
    name = (name or "").strip()
    if not name:
        raise ValueError("smart crate needs a name")
    con = connect(db_path)
    con.execute(
        "INSERT INTO smart_crates(name, spec, created) VALUES(?,?,?) "
        "ON CONFLICT(name) DO UPDATE SET spec=excluded.spec",
        (name, json.dumps(spec), time.time()))
    con.commit()
    con.close()


def list_smart_crates(db_path: Path = DB_PATH) -> list[str]:
    con = connect(db_path)
    rows = con.execute("SELECT name FROM smart_crates ORDER BY name COLLATE NOCASE").fetchall()
    con.close()
    return [r[0] for r in rows]


def read_smart_crate(name: str, db_path: Path = DB_PATH) -> dict | None:
    con = connect(db_path)
    r = con.execute("SELECT spec FROM smart_crates WHERE name=?", (name,)).fetchone()
    con.close()
    if not r:
        return None
    try:
        return json.loads(r[0])
    except (TypeError, ValueError):
        return None


def delete_smart_crate(name: str, db_path: Path = DB_PATH) -> None:
    con = connect(db_path)
    con.execute("DELETE FROM smart_crates WHERE name=?", (name,))
    con.commit()
    con.close()


# --- library health (duplicates / missing / low quality) --------------------
LOSSLESS_EXTS = {".flac", ".wav", ".aiff", ".aif", ".alac"}


def _norm_text(s: str | None) -> str:
    """Lowercase + collapse whitespace for duplicate grouping (so 'Charli XCX' == 'charli xcx')."""
    import re
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def estimate_kbps(track: Track) -> float | None:
    """Rough average bitrate from file size / duration (kbps), or None if unknowable."""
    if not track.duration or track.duration <= 0 or not track.size:
        return None
    return (track.size * 8) / track.duration / 1000.0


def _quality_rank(track: Track):
    """Sort key (higher = better copy to KEEP): a file that EXISTS beats a missing/phantom one
    (so a stale path is never the suggested keep), then lossless beats lossy, then bitrate, size.
    Guards against the SMB case-folder trap where a phantom path would otherwise win on a tie."""
    exists = Path(track.path).exists()
    lossless = Path(track.path).suffix.lower() in LOSSLESS_EXTS
    return (1 if exists else 0, 1 if lossless else 0, estimate_kbps(track) or 0.0, track.size or 0)


def find_duplicate_groups(db_path: Path = DB_PATH) -> list[list[Track]]:
    """Groups of tracks that are the same recording by normalized (artist, title). Each group has
    >1 member, sorted best-copy-first (the suggested KEEP), so the rest are redundant. Catches the
    classic 'same song as both FLAC and MP3' across buckets."""
    groups: dict[tuple, list[Track]] = {}
    for t in search(limit=100000, db_path=db_path):
        title = _norm_text(t.title)
        if not title:
            continue                                  # untitled rows are unreliable to group
        groups.setdefault((_norm_text(t.artist), title), []).append(t)
    out = [ts for ts in groups.values() if len(ts) > 1]
    for ts in out:
        ts.sort(key=_quality_rank, reverse=True)
    out.sort(key=lambda g: (-len(g), _norm_text(g[0].artist), _norm_text(g[0].title)))
    return out


def find_missing_files(db_path: Path = DB_PATH) -> list[Track]:
    """Indexed tracks whose file no longer exists on disk (stat over the library mount)."""
    return [t for t in search(limit=100000, db_path=db_path) if not Path(t.path).exists()]


def find_low_quality(db_path: Path = DB_PATH, min_kbps: float = 256.0) -> list[tuple[Track, int]]:
    """Lossy files below min_kbps, as (Track, kbps) — candidates to re-grab in better quality."""
    out = []
    for t in search(limit=100000, db_path=db_path):
        if Path(t.path).suffix.lower() in LOSSLESS_EXTS:
            continue
        kbps = estimate_kbps(t)
        if kbps is not None and kbps < min_kbps:
            out.append((t, round(kbps)))
    out.sort(key=lambda x: x[1])
    return out


def library_health(db_path: Path = DB_PATH, min_kbps: float = 256.0) -> dict:
    """One-shot health report: duplicate groups, missing files, low-quality lossy copies."""
    dups = find_duplicate_groups(db_path)
    return {
        "duplicate_groups": dups,
        "redundant_copies": sum(len(g) - 1 for g in dups),   # how many files could be removed
        "missing": find_missing_files(db_path),
        "low_quality": find_low_quality(db_path, min_kbps),
    }


# --- export -----------------------------------------------------------------
def _sanitize(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("-" if c in bad else c for c in name).strip().strip(".")
    return out or "crate"


# --- harmonic mixing (Camelot) + ratings ----------------------------------
def _parse_camelot(code):
    if not code or len(code) < 2:
        return None
    try:
        num, letter = int(code[:-1]), code[-1].upper()
    except ValueError:
        return None
    if letter not in ("A", "B") or not (1 <= num <= 12):
        return None
    return num, letter


def camelot_neighbors(code: str) -> dict:
    """Harmonically-compatible Camelot codes -> relationship label, for a seed key."""
    p = _parse_camelot(code)
    if not p:
        return {}
    num, letter = p
    wrap = lambda n: (n - 1) % 12 + 1
    other = "B" if letter == "A" else "A"
    return {
        f"{num}{letter}": "same key",
        f"{num}{other}": "relative major/minor",
        f"{wrap(num - 1)}{letter}": "-1 (energy down)",
        f"{wrap(num + 1)}{letter}": "+1 (energy up)",
    }


def harmonic_matches(seed: Track, db_path: Path = DB_PATH,
                     bpm_tol: float = 0.08, limit: int = 300) -> list[Track]:
    """Owned tracks that mix well with `seed`: harmonically-compatible key AND BPM within
    ±bpm_tol (also accepting half/double-time). The key+BPM gate is the hard constraint; within
    it, ranking is sonic-aware — candidates are ordered by the fused `mixability` score (key +
    tempo + sonic vibe), so harmonically-equal tracks that actually *sound* alike rank first.
    Falls back to key-rank/BPM-distance order when no sonic vectors are present."""
    if not seed.key or not seed.bpm:
        return []
    neigh = camelot_neighbors(seed.key)
    if not neigh:
        return []
    con = connect(db_path)
    rows = con.execute(
        f"SELECT * FROM tracks WHERE key IN ({','.join('?' * len(neigh))}) "
        f"AND path != ? AND bpm IS NOT NULL AND bpm > 0",
        (*neigh.keys(), seed.path)).fetchall()
    con.close()
    b = seed.bpm
    order = {"same key": 0, "relative major/minor": 1, "-1 (energy down)": 2, "+1 (energy up)": 3}

    def bpm_dist(x):
        best = None
        for target in (b, b / 2, b * 2):
            d = abs(x - target) / target
            if d <= bpm_tol and (best is None or d < best):
                best = d
        return best

    vecs = load_vectors()           # cached snapshot of the full 512-d sonic vectors (may be empty)
    secs = load_section_vectors()   # cached intro/outro vectors for transition flow (may be empty)
    scored = []
    for r in rows:
        d = bpm_dist(r["bpm"])
        if d is None:
            continue
        t = _row_to_track(r)
        # higher = better; -mixability so it sorts ascending alongside the key/bpm tiebreakers
        mix = mixability(seed, t, vectors=vecs, sections=secs)
        scored.append((-mix, order.get(neigh.get(r["key"], ""), 9), d, t))
    scored.sort(key=lambda s: (s[0], s[1], s[2]))
    return [t for _, _, _, t in scored[:limit]]


# --- Sonic similarity + mixability (full 512-d vectors) ---------------------
# The MAP is a 2D *display* projection; recommendation math must run on the full vectors.
# We snapshot music_vectors.sqlite (written by embed_muq.py) once, normalize, and cache an
# {local_path: vec} dict + a stacked matrix for fast cosine. Cheap (~613x512 floats).
_VEC_CACHE: dict | None = None         # {local_path(str): np.ndarray(float32, L2-normed)}
_VEC_MTIME: float | None = None        # source mtime the cache was built from
_CLU_CACHE: dict | None = None         # {local_path(str): int cluster_id}
_CLU_MTIME: float | None = None
_SEC_CACHE: dict | None = None         # {local_path(str): (intro_unit, outro_unit)} for transitions
_SEC_MTIME: float | None = None


def clear_vector_cache() -> None:
    global _VEC_CACHE, _VEC_MTIME, _CLU_CACHE, _CLU_MTIME, _SEC_CACHE, _SEC_MTIME
    _VEC_CACHE = None
    _VEC_MTIME = None
    _CLU_CACHE = None
    _CLU_MTIME = None
    _SEC_CACHE = None
    _SEC_MTIME = None


def load_vectors(vectors_path: Path = None, lib_root: Path = None, force: bool = False) -> dict:
    """Return {local_path: 512-d float32 unit vector} from music_vectors.sqlite, cached.

    Keyed on the SAME local Windows path as the tracks table (lib_root / relpath), so callers can
    look up a Track's vector by `track.path`. Empty dict if the sidecar isn't present yet.
    """
    global _VEC_CACHE, _VEC_MTIME
    vectors_path = Path(vectors_path) if vectors_path else VECTORS_PATH
    lib_root = Path(lib_root) if lib_root else LIB_ROOT
    if not vectors_path.exists():
        return _VEC_CACHE or {}
    try:
        mt = vectors_path.stat().st_mtime
    except OSError:
        mt = None
    if not force and _VEC_CACHE is not None and mt == _VEC_MTIME:
        return _VEC_CACHE
    try:
        import numpy as np
        import tempfile
        tmp = Path(tempfile.gettempdir()) / "crate_vectors_snapshot.sqlite"
        shutil.copy2(vectors_path, tmp)  # snapshot to dodge SMB read-locks while the box writes
        vcon = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        rows = vcon.execute("SELECT relpath, vec FROM vectors").fetchall()
        # MEAN-CENTER at read time with the SAME dataset-mean the box UMAP was built with (persisted
        # in the vector_stats table by embed_muq.py / vectors.recompute_and_store). Raw sonic vectors
        # can be anisotropic (high cosine between everything) so un-centered similarity is noise; this
        # keeps similar_tracks / mixability in step with the map. Absent (old sidecar) -> raw, as before.
        try:
            mr = vcon.execute("SELECT v FROM vector_stats WHERE k='mean'").fetchone()
            mean = np.frombuffer(mr[0], dtype="float32").astype("float64") if mr and mr[0] else None
        except Exception:
            mean = None
        vcon.close()
        cache: dict = {}
        for rel, blob in rows:
            if not blob:
                continue
            v = np.frombuffer(blob, dtype="float32").astype("float64")
            if mean is not None:
                v = v - mean
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                continue
            cache[str(Path(lib_root) / rel)] = (v / n).astype("float32")   # unit -> cosine = dot
        _VEC_CACHE = cache
        _VEC_MTIME = mt
        return cache
    except Exception:
        return _VEC_CACHE or {}


def load_clusters(clusters_path: Path = None, lib_root: Path = None, force: bool = False) -> dict:
    """Return {local_path: cluster_id} from clusters.sqlite, cached.

    Keyed on the SAME local Windows path as the tracks table (lib_root / relpath), so callers can
    look up a Track's cluster by `track.path`. cluster_id -1 means noise / unclustered. Empty dict
    if the sidecar isn't present yet.
    """
    global _CLU_CACHE, _CLU_MTIME
    clusters_path = Path(clusters_path) if clusters_path else CLUSTERS_PATH
    lib_root = Path(lib_root) if lib_root else LIB_ROOT
    if not clusters_path.exists():
        return _CLU_CACHE or {}
    try:
        mt = clusters_path.stat().st_mtime
    except OSError:
        mt = None
    if not force and _CLU_CACHE is not None and mt == _CLU_MTIME:
        return _CLU_CACHE
    try:
        import tempfile
        tmp = Path(tempfile.gettempdir()) / "crate_clusters_snapshot.sqlite"
        shutil.copy2(clusters_path, tmp)
        ccon = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        try:
            rows = ccon.execute("SELECT relpath, cluster_id FROM clusters").fetchall()
        except sqlite3.OperationalError:
            ccon.close()
            _CLU_CACHE, _CLU_MTIME = {}, mt
            return _CLU_CACHE
        ccon.close()
        cache = {str(Path(lib_root) / rel): int(cluster_id) for rel, cluster_id in rows}
        _CLU_CACHE = cache
        _CLU_MTIME = mt
        return cache
    except Exception:
        return _CLU_CACHE or {}


def sonic_similarity(path_a: str, path_b: str, vectors: dict | None = None) -> float | None:
    """Cosine similarity (0..1, clamped) between two tracks' sonic (MuQ-MuLan) vectors, or None if
    either is un-embedded. Vectors are unit-normalized so cosine = dot."""
    import numpy as np
    vecs = vectors if vectors is not None else load_vectors()
    a, b = vecs.get(path_a), vecs.get(path_b)
    if a is None or b is None:
        return None
    return max(0.0, min(1.0, float(np.dot(a, b))))


def load_section_vectors(vectors_path: Path = None, lib_root: Path = None, force: bool = False) -> dict:
    """Return {local_path: (intro_unit, outro_unit)} from music_vectors.sqlite, cached. These are the
    per-end sonic windows (vec_intro near the start, vec_outro near the end) used for TRANSITION
    matching: a track's OUTRO vs a candidate's INTRO says how the *mix point* flows, not just whether
    the two tracks sound alike overall. Mean-centered + unit-normed with the same dataset-mean as the
    whole-track vectors (so cosine is meaningful). Empty if the sidecar lacks the section columns
    (older analysis runs) or isn't present yet."""
    global _SEC_CACHE, _SEC_MTIME
    vectors_path = Path(vectors_path) if vectors_path else VECTORS_PATH
    lib_root = Path(lib_root) if lib_root else LIB_ROOT
    if not vectors_path.exists():
        return _SEC_CACHE or {}
    try:
        mt = vectors_path.stat().st_mtime
    except OSError:
        mt = None
    if not force and _SEC_CACHE is not None and mt == _SEC_MTIME:
        return _SEC_CACHE
    try:
        import numpy as np
        import tempfile
        tmp = Path(tempfile.gettempdir()) / "crate_vectors_snapshot.sqlite"
        shutil.copy2(vectors_path, tmp)
        vcon = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        try:
            rows = vcon.execute("SELECT relpath, vec_intro, vec_outro FROM vectors").fetchall()
        except sqlite3.OperationalError:        # old sidecar without section columns
            vcon.close()
            _SEC_CACHE, _SEC_MTIME = {}, mt
            return _SEC_CACHE
        try:
            mr = vcon.execute("SELECT v FROM vector_stats WHERE k='mean'").fetchone()
            mean = np.frombuffer(mr[0], dtype="float32").astype("float64") if mr and mr[0] else None
        except Exception:
            mean = None
        vcon.close()

        def _unit(blob):
            if not blob:
                return None
            v = np.frombuffer(blob, dtype="float32").astype("float64")
            if mean is not None and v.shape == mean.shape:
                v = v - mean
            n = float(np.linalg.norm(v))
            return (v / n).astype("float32") if n >= 1e-9 else None

        cache: dict = {}
        for rel, ib, ob in rows:
            iv, ov = _unit(ib), _unit(ob)
            if iv is not None or ov is not None:
                cache[str(Path(lib_root) / rel)] = (iv, ov)
        _SEC_CACHE, _SEC_MTIME = cache, mt
        return cache
    except Exception:
        return _SEC_CACHE or {}


def transition_score(path_a: str, path_b: str, sections: dict | None = None) -> float | None:
    """How well A flows INTO B at the mix point: cosine (0..1) of A's OUTRO vector vs B's INTRO
    vector. Directional (a->b != b->a). None if either section vector is missing."""
    import numpy as np
    secs = sections if sections is not None else load_section_vectors()
    a, b = secs.get(path_a), secs.get(path_b)
    if not a or not b:
        return None
    out_a, in_b = a[1], b[0]
    if out_a is None or in_b is None:
        return None
    return max(0.0, min(1.0, float(np.dot(out_a, in_b))))


def similar_tracks(seed: Track, n: int = 50, db_path: Path = DB_PATH,
                   vectors_path: Path = None, lib_root: Path = None) -> list[tuple[Track, float]]:
    """Tracks most sonically similar to `seed` by full-512-d sonic cosine (vibe/timbre), best-first.
    Returns [(Track, similarity 0..1)]. Empty if the seed isn't embedded or no vectors exist."""
    import numpy as np
    vecs = load_vectors(vectors_path, lib_root)
    seed_vec = vecs.get(seed.path)
    if seed_vec is None or len(vecs) < 2:
        return []
    # music_vectors.sqlite can hold MORE rows than the live index (stale/un-indexed files), so
    # restrict to currently-indexed tracks BEFORE ranking — otherwise the top-N can be all
    # un-indexed vectors that then filter out to nothing.
    con = connect(db_path)
    rows = {r["path"]: _row_to_track(r) for r in con.execute("SELECT * FROM tracks")}
    con.close()
    paths = [p for p in vecs if p != seed.path and p in rows]
    if not paths:
        return []
    mat = np.stack([vecs[p] for p in paths])           # (N, 512), unit rows
    sims = mat @ seed_vec                               # cosine, since both unit
    order = np.argsort(-sims)[:n]
    return [(rows[paths[i]], max(0.0, min(1.0, float(sims[i])))) for i in order]


def _bpm_score(a: float | None, b: float | None, tol: float = 0.08) -> float:
    """0..1 BPM compatibility, accepting half/double-time. 1 = identical tempo, 0 = beyond tol."""
    if not a or not b or a <= 0 or b <= 0:
        return 0.0
    best = None
    for target in (b, b / 2, b * 2):
        d = abs(a - target) / target
        if best is None or d < best:
            best = d
    return max(0.0, 1.0 - best / tol) if best is not None and best <= tol else 0.0


def _key_score(a: str | None, b: str | None) -> float:
    """0..1 harmonic compatibility from Camelot codes: 1 same key, ~0.8 relative/±1, else 0."""
    if not a or not b:
        return 0.0
    neigh = camelot_neighbors(a)
    if b == a:
        return 1.0
    rel = neigh.get(b)
    if rel == "relative major/minor":
        return 0.85
    if rel in ("-1 (energy down)", "+1 (energy up)"):
        return 0.8
    return 0.0


def compat_penalty(a: Track, b: Track) -> float:
    """0..1 soft DJ-compatibility penalty: 0 = fully mixable, 1 = maximally incompatible."""
    return 1.0 - 0.5 * _bpm_score(a.bpm, b.bpm) - 0.5 * _key_score(a.key, b.key)


def mixability(a: Track, b: Track, vectors: dict | None = None, sections: dict | None = None,
               weights: tuple[float, float, float, float] = (0.34, 0.30, 0.20, 0.16)) -> float:
    """Fused 0..1 DJ-mixability of A into B, blending four factors (weights = key, tempo, sound,
    transition):
      • key        — Camelot harmonic compatibility (a hard mixing constraint)
      • tempo      — BPM proximity, accepting half/double-time (the other hard constraint)
      • sound      — whole-track sonic cosine: do they share a vibe/timbre
      • transition — A's OUTRO vs B's INTRO: does the mix POINT actually flow (directional)
    Any factor whose data is missing (un-embedded track, no section vectors, old sidecar) is dropped
    and its weight redistributed over the rest, so the score stays a clean 0..1 either way. Note the
    transition term makes this DIRECTIONAL — mixability(a,b) is 'how well a leads into b'.
    """
    wk, wb, wc, wt = weights
    key_s = _key_score(a.key, b.key)
    bpm_s = _bpm_score(a.bpm, b.bpm)
    sonic_s = sonic_similarity(a.path, b.path, vectors)
    trans_s = transition_score(a.path, b.path, sections)
    terms = [(wk, key_s), (wb, bpm_s)]
    if sonic_s is not None:
        terms.append((wc, sonic_s))
    if trans_s is not None:
        terms.append((wt, trans_s))
    tot = sum(w for w, _ in terms)
    return sum(w * s for w, s in terms) / tot if tot else 0.0


def compatible_next(seed: Track, exclude_paths=(), db_path: Path = DB_PATH,
                    limit: int = 200) -> list[tuple[Track, float]]:
    """Ranked owned tracks that mix well coming AFTER `seed`, best-first, each with its 0..1
    mixability score — the building block of the step-by-step CHAIN (you pick the next track, it
    re-ranks from there). Same key+BPM gate as the COMPATIBLE list, ranked by the fused mixability
    (key+tempo+sound+transition). `exclude_paths` drops tracks already in the chain."""
    ex = set(exclude_paths) | {seed.path}
    cands = harmonic_matches(seed, db_path=db_path, limit=limit + len(ex))
    vecs = load_vectors()
    secs = load_section_vectors()
    out = [(t, mixability(seed, t, vectors=vecs, sections=secs))
           for t in cands if t.path not in ex]
    return out[:limit]


def _energy_pick(cur: Track, cands: list[tuple[Track, float]], energy: str) -> Track:
    """Choose the next track from ranked `cands` honoring an energy arc: 'up' prefers the
    best-mixing candidate that doesn't drop energy, 'down' the reverse; falls back to the top
    mixability pick when nothing fits the direction. 'flat' just takes the best mix."""
    if energy == "flat":
        return cands[0][0]
    ce = cur.energy or 0.0
    if energy == "up":
        pref = [t for t, _ in cands if (t.energy or 0.0) >= ce - 1e-6]
    else:                                          # 'down'
        pref = [t for t, _ in cands if (t.energy or 0.0) <= ce + 1e-6]
    return pref[0] if pref else cands[0][0]


def build_path(seed: Track, length: int = 12, db_path: Path = DB_PATH,
               energy: str = "flat", topk: int = 8) -> list[Track]:
    """Draft a DJ set as an ordered path starting from `seed`: greedily append the owned track that
    best mixes from the current last track (never repeating), optionally shaping an energy arc
    ('flat' | 'up' | 'down'). This is a STARTING SKETCH the DJ then edits — not a finished set.
    Stops early if it runs out of compatible tracks. Returns [Track] (seed first)."""
    chosen = [seed]
    used = {seed.path}
    cur = seed
    for _ in range(max(0, length - 1)):
        cands = compatible_next(cur, used, db_path=db_path, limit=topk)
        if not cands:
            break
        pick = _energy_pick(cur, cands, energy)
        chosen.append(pick)
        used.add(pick.path)
        cur = pick
    return chosen


_SNAP_CACHE: dict[str, dict] = {}   # tag -> {"mtime": float} for mtime-cached sidecar snapshots


def _snapshot(src: Path, tag: str):
    """Local, mtime-cached snapshot of a sidecar sqlite that lives on the SMB share (Z:).

    Copying to temp avoids SMB read-locks / partial reads while the box is still writing, but the old
    code re-copied on EVERY call — so toggling 2D<->3D or ARTISTS re-copied the file over the network
    each time. Here we re-copy only when the source mtime changes (the `_waveform_snapshot` pattern),
    so repeated MAP / mix-brain interactions cost zero network I/O after the first. Returns the local
    snapshot path, or None if the source is missing / unreadable.
    """
    import tempfile
    src = Path(src)
    try:
        if not src.exists():
            return None
        mt = src.stat().st_mtime
        tmp = Path(tempfile.gettempdir()) / f"crate_{tag}_snapshot.sqlite"
        ent = _SNAP_CACHE.get(tag)
        if ent is None or ent.get("mtime") != mt or not tmp.exists():
            shutil.copy2(src, tmp)
            _SNAP_CACHE[tag] = {"mtime": mt}
        return tmp
    except Exception:
        return None


def _tracks_by_path(con) -> dict:
    """All tracks keyed by path in ONE query — so the *_with_coords joins don't run a SELECT per map
    point (an N+1 that was one SMB-free but still per-row round-trip for every dot on the galaxy)."""
    return {r["path"]: r for r in con.execute("SELECT * FROM tracks").fetchall()}


def tracks_with_coords(db_path: Path = DB_PATH, umap_path: Path = UMAP_PATH,
                       lib_root: Path = LIB_ROOT) -> list[tuple[Track, float, float]]:
    """Join UMAP coords (from the box) with local track metadata for the MAP view.
    Returns [(Track, x, y)] with x,y in [0,1]. Empty if the map hasn't been built yet."""
    tmp = _snapshot(umap_path, "umap")
    if tmp is None:
        return []
    try:
        ucon = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        coords = ucon.execute("SELECT relpath, x, y FROM coords").fetchall()
        ucon.close()
    except Exception:
        return []
    con = connect(db_path)
    by_path = _tracks_by_path(con)
    con.close()
    out: list[tuple[Track, float, float]] = []
    for rel, x, y in coords:
        r = by_path.get(str(Path(lib_root) / rel))
        if r:
            out.append((_row_to_track(r), x, y))
    return out


def tracks_with_coords3d(db_path: Path = DB_PATH, umap_path: Path = UMAP_PATH,
                         lib_root: Path = LIB_ROOT) -> list[tuple[Track, float, float, float]]:
    """Like tracks_with_coords but the 3D PaCMAP positions (the `coords3d` table umap_music.py writes
    alongside the 2D `coords`). Returns [(Track, x, y, z)] with x,y,z in [0,1]; empty if the 3D map
    hasn't been built (older sidecars only have 2D — the caller should fall back to tracks_with_coords)."""
    tmp = _snapshot(umap_path, "umap3d")
    if tmp is None:
        return []
    try:
        ucon = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        has3d = ucon.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='coords3d'").fetchone()
        coords = ucon.execute("SELECT relpath, x, y, z FROM coords3d").fetchall() if has3d else []
        ucon.close()
    except Exception:
        return []
    con = connect(db_path)
    by_path = _tracks_by_path(con)
    con.close()
    out: list[tuple[Track, float, float, float]] = []
    for rel, x, y, z in coords:
        r = by_path.get(str(Path(lib_root) / rel))
        if r:
            out.append((_row_to_track(r), x, y, z))
    return out


def artists_with_coords(artist_umap_path: Path = ARTIST_UMAP_PATH) -> list[tuple[str, float, float, int]]:
    """[(artist, x, y, n)] from the ARTIST-level UMAP sidecar (umap_artists.py); x,y in [0,1],
    n = tracks that fed the artist vector. Empty if the artist map hasn't been built yet."""
    tmp = _snapshot(artist_umap_path, "artist_umap")
    if tmp is None:
        return []
    try:
        c = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        rows = c.execute("SELECT artist, x, y, n FROM artists").fetchall()
        c.close()
        return [(r[0], float(r[1]), float(r[2]), int(r[3])) for r in rows]
    except Exception:
        return []


def artists_with_coords3d(artist_umap_path: Path = ARTIST_UMAP_PATH) -> list[tuple[str, float, float, float, int]]:
    """[(artist, x, y, z, n)] from the 3D ARTIST UMAP (`artists3d` table umap_artists.py writes
    alongside the 2D `artists`). x,y,z in [0,1]. Empty if the 3D artist map hasn't been built
    (older sidecars only have 2D — the caller should fall back to the 2D artist view)."""
    tmp = _snapshot(artist_umap_path, "artist_umap3d")
    if tmp is None:
        return []
    try:
        c = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        has3d = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='artists3d'").fetchone()
        rows = c.execute("SELECT artist, x, y, z, n FROM artists3d").fetchall() if has3d else []
        c.close()
        return [(r[0], float(r[1]), float(r[2]), float(r[3]), int(r[4])) for r in rows]
    except Exception:
        return []


_WF_SNAP = {"mtime": None, "path": None}   # cache the waveform-sidecar snapshot across lookups


def _waveform_snapshot(waveform_path: Path):
    """Snapshot the box's waveforms.sqlite locally (avoids SMB read-locks while the box writes);
    re-copy only when the source mtime changes. Returns the local snapshot path or None."""
    import tempfile
    src = Path(waveform_path)
    if not src.exists():
        return None
    try:
        mt = src.stat().st_mtime
        if _WF_SNAP["path"] is None or _WF_SNAP["mtime"] != mt:
            tmp = Path(tempfile.gettempdir()) / "crate_waveforms_snapshot.sqlite"
            shutil.copy2(src, tmp)
            _WF_SNAP.update(mtime=mt, path=tmp)
        return _WF_SNAP["path"]
    except Exception:
        return None


def get_waveform(path: str, bins: int = 1600, lib_root: Path = LIB_ROOT,
                 waveform_path: Path = WAVEFORM_PATH):
    """Return an (N,3) uint8 numpy array (low/mid/high energy per column) for `path`, or None.

    Prefers the box-precomputed colored sidecar (keyed by relpath under lib_root). Falls back to a
    flat PC-side decode via soundfile for tracks the box hasn't analyzed (same value in all 3 bands).
    """
    import numpy as np
    try:
        rel = Path(path).relative_to(lib_root).as_posix()
    except ValueError:
        rel = None
    if rel:
        snap = _waveform_snapshot(waveform_path)
        if snap is not None:
            try:
                wcon = sqlite3.connect(f"file:{snap}?mode=ro", uri=True)
                row = wcon.execute(
                    "SELECT bins, data FROM waveforms WHERE relpath=?", (rel,)).fetchone()
                wcon.close()
                if row and row[1]:
                    return np.frombuffer(row[1], np.uint8).reshape(row[0], 3)
            except Exception:
                pass
    return _waveform_local(path, bins)


def _waveform_local(path: str, bins: int = 1600):
    """Fallback: decode the track on the PC (soundfile) into a flat amplitude waveform."""
    try:
        import numpy as np
        import soundfile as sf
        info = sf.info(str(path))
        if not info.frames:
            return None
        block = max(1, info.frames // bins)
        peaks = []
        for b in sf.blocks(str(path), blocksize=block, dtype="float32", fill_value=0):
            m = np.abs(b).max(axis=1) if b.ndim > 1 else np.abs(b)
            peaks.append(float(m.max()) if len(m) else 0.0)
        arr = np.array(peaks[:bins], dtype=np.float64)
        if arr.size == 0 or arr.max() <= 0:
            return None
        u8 = np.clip((arr / arr.max()) ** 0.7 * 255.0, 0, 255).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=1)   # flat = same level in all 3 bands
    except Exception:
        return None


def set_rating(path: str, rating: int | None, db_path: Path = DB_PATH) -> None:
    con = connect(db_path)
    con.execute("UPDATE tracks SET rating=? WHERE path=?", (rating or None, path))
    con.commit()
    con.close()


def get_track_tags(path: str, db_path: Path = DB_PATH) -> dict[str, list[str]]:
    con = connect(db_path)
    rows = con.execute(
        "SELECT category, value FROM track_tags WHERE path=? "
        "ORDER BY category COLLATE NOCASE, value COLLATE NOCASE",
        (path,)).fetchall()
    con.close()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["category"], []).append(r["value"])
    return out


def all_tag_summaries(db_path: Path = DB_PATH) -> dict[str, str]:
    """{path: 'tag1, tag2, …'} across all categories — a compact column for the table.
    One query for the whole library (cheap), grouped in Python."""
    con = connect(db_path)
    rows = con.execute(
        "SELECT path, value FROM track_tags ORDER BY value COLLATE NOCASE").fetchall()
    con.close()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["path"], []).append(r["value"])
    return {p: ", ".join(v) for p, v in out.items()}


def cue_counts(db_path: Path = DB_PATH) -> dict[str, int]:
    """{path: number of cues} for the whole library — a compact column for the table."""
    con = connect(db_path)
    rows = con.execute("SELECT path, COUNT(*) FROM track_cues GROUP BY path").fetchall()
    con.close()
    return {r[0]: r[1] for r in rows}


def set_track_tags(path: str, category: str, values: list[str],
                   db_path: Path = DB_PATH) -> None:
    con = connect(db_path)
    con.execute("DELETE FROM track_tags WHERE path=? AND category=?", (path, category))
    seen: set[str] = set()
    for value in values:
        v = value.strip()
        if not v or v in seen:
            continue
        seen.add(v)
        con.execute("INSERT INTO track_tags(path,category,value) VALUES(?,?,?)",
                    (path, category, v))
    con.commit()
    con.close()


def add_track_tag(path: str, category: str, value: str,
                  db_path: Path = DB_PATH) -> None:
    v = value.strip()
    if not v:
        return
    con = connect(db_path)
    con.execute("INSERT OR IGNORE INTO track_tags(path,category,value) VALUES(?,?,?)",
                (path, category, v))
    con.commit()
    con.close()


def remove_track_tag(path: str, category: str, value: str,
                     db_path: Path = DB_PATH) -> None:
    con = connect(db_path)
    con.execute("DELETE FROM track_tags WHERE path=? AND category=? AND value=?",
                (path, category, value))
    con.commit()
    con.close()


def all_tag_values(category: str, db_path: Path = DB_PATH) -> list[str]:
    con = connect(db_path)
    rows = con.execute(
        "SELECT DISTINCT value FROM track_tags WHERE category=? ORDER BY value COLLATE NOCASE",
        (category,)).fetchall()
    con.close()
    return [r["value"] for r in rows]


def set_color(path: str, color: str | None, db_path: Path = DB_PATH) -> None:
    con = connect(db_path)
    con.execute("UPDATE tracks SET color=? WHERE path=?", ((color or None), path))
    con.commit()
    con.close()


def set_comment(path: str, comment: str | None, db_path: Path = DB_PATH) -> None:
    con = connect(db_path)
    con.execute("UPDATE tracks SET comment=? WHERE path=?", ((comment or None), path))
    con.commit()
    con.close()


def add_cue(path: str, kind: str, idx: str, position_ms: int,
            color: str | None = None, name: str | None = None,
            db_path: Path = DB_PATH) -> int:
    con = connect(db_path)
    cur = con.execute(
        "INSERT INTO track_cues(path,kind,idx,position_ms,color,name) VALUES(?,?,?,?,?,?)",
        (path, kind, idx, position_ms, color, name))
    con.commit()
    cue_id = int(cur.lastrowid)
    con.close()
    return cue_id


def get_cues(path: str, db_path: Path = DB_PATH) -> list[dict]:
    con = connect(db_path)
    rows = con.execute(
        "SELECT id, kind, idx, position_ms, color, name FROM track_cues WHERE path=? "
        "ORDER BY position_ms",
        (path,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


def delete_cue(cue_id: int, db_path: Path = DB_PATH) -> None:
    con = connect(db_path)
    con.execute("DELETE FROM track_cues WHERE id=?", (cue_id,))
    con.commit()
    con.close()


def clear_cues(path: str, db_path: Path = DB_PATH) -> None:
    con = connect(db_path)
    con.execute("DELETE FROM track_cues WHERE path=?", (path,))
    con.commit()
    con.close()


def update_cue(cue_id: int, position_ms: int, db_path: Path = DB_PATH) -> None:
    """Move a cue to a new position (used when dragging a flag on the waveform)."""
    con = connect(db_path)
    con.execute("UPDATE track_cues SET position_ms=? WHERE id=?", (int(position_ms), cue_id))
    con.commit()
    con.close()


def delete_tracks(track_paths: list[str], db_path: Path = DB_PATH,
                  quarantine: Path = QUARANTINE, lib_root: Path = LIB_ROOT) -> dict:
    """Reversible delete: MOVE each track into the quarantine folder (preserving bucket/artist
    structure) and drop it from the index. Nothing is hard-deleted — removal goes to
    <lib_root>/.crate/trash, from where restore_tracks() can put it back.

    Returns {moved, failed:[(path,err)], quarantine}.
    """
    con = connect(db_path)
    moved = 0
    failed: list[tuple[str, str]] = []
    qroot = Path(quarantine)
    for p in track_paths:
        src = Path(p)
        try:
            if src.exists():
                try:
                    rel = src.relative_to(lib_root)
                except ValueError:
                    rel = Path(src.name)
                dest = qroot / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    dest = dest.with_name(f"{dest.stem}_{int(time.time())}{dest.suffix}")
                shutil.move(str(src), str(dest))
                # remember EXACTLY where it came from so restore can put it back even if `src` was
                # outside lib_root (an added scan root), where `rel` is only a basename.
                con.execute(
                    "INSERT OR REPLACE INTO quarantine(trash_rel, orig_path, deleted) VALUES(?,?,?)",
                    (dest.relative_to(qroot).as_posix(), str(src), time.time()))
            con.execute("DELETE FROM tracks WHERE path=?", (p,))
            moved += 1
        except Exception as e:
            failed.append((p, f"{type(e).__name__}: {e}"))
    con.commit()
    con.close()
    return {"moved": moved, "failed": failed, "quarantine": str(qroot)}


def list_quarantine(quarantine: Path = QUARANTINE) -> list[dict]:
    """Everything currently in the trash, newest-first. Each: relpath/name/bucket/artist/size/
    mtime/abspath (bucket+artist parsed from the preserved <bucket>/<artist>/<file> structure)."""
    qroot = Path(quarantine)
    if not qroot.exists():
        return []
    out: list[dict] = []
    for p in qroot.rglob("*"):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rel = p.relative_to(qroot)
        parts = rel.as_posix().split("/")
        out.append({"relpath": rel.as_posix(), "name": p.name,
                    "bucket": parts[0] if len(parts) > 1 else "",
                    "artist": parts[-2] if len(parts) >= 2 else "",
                    "size": st.st_size, "mtime": st.st_mtime, "abspath": str(p)})
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


def _contained(root: Path, rel: str) -> Path | None:
    """Resolve `rel` under `root` and return it ONLY if it stays inside `root`. Guards the destructive
    trash ops: a `..` segment OR an ABSOLUTE `rel` (`qroot / "/etc/x"` == `/etc/x` in pathlib — the
    left side is discarded!) would otherwise unlink/move a file outside the trash."""
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        return None
    root = Path(root).resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def restore_tracks(relpaths: list[str], quarantine: Path = QUARANTINE,
                   lib_root: Path = LIB_ROOT, db_path: Path = DB_PATH) -> dict:
    """Move files back from the trash into the library and re-index just those files so they
    reappear. Returns {restored, failed:[(relpath,err)]}."""
    qroot = Path(quarantine)
    con = connect(db_path)
    restored = 0
    failed: list[tuple[str, str]] = []
    for rel in relpaths:
        src = _contained(qroot, rel)                # source must stay inside the trash root...
        if src is None:
            failed.append((rel, "unsafe path"))
            continue
        # restore to the EXACT original location when we recorded one (handles deletes from added
        # scan roots outside lib_root); otherwise fall back to lib_root/<rel>, contained.
        orig = con.execute("SELECT orig_path FROM quarantine WHERE trash_rel=?", (rel,)).fetchone()
        if orig and orig[0] and Path(orig[0]).is_absolute():
            dest = Path(orig[0])
        else:
            dest = _contained(Path(lib_root), rel)
        if dest is None:
            failed.append((rel, "unsafe path"))
            continue
        try:
            if not src.exists():
                failed.append((rel, "not in trash"))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest = dest.with_name(f"{dest.stem}_{int(time.time())}{dest.suffix}")
            shutil.move(str(src), str(dest))
            con.execute("DELETE FROM quarantine WHERE trash_rel=?", (rel,))
            st = dest.stat()
            bucket = rel.split("/")[0] if "/" in rel else ""
            artist, title, album, duration, bpm = read_tags(dest)
            con.execute(
                """INSERT INTO tracks(path,bucket,artist,title,album,ext,size,mtime,duration,bpm,key)
                   VALUES(?,?,?,?,?,?,?,?,?,?,NULL)
                   ON CONFLICT(path) DO UPDATE SET bucket=excluded.bucket, artist=excluded.artist,
                     title=excluded.title, album=excluded.album, size=excluded.size,
                     mtime=excluded.mtime, duration=excluded.duration""",
                (str(dest), bucket, artist, title, album, dest.suffix.lower(),
                 st.st_size, st.st_mtime, duration, bpm))
            restored += 1
        except Exception as e:
            failed.append((rel, f"{type(e).__name__}: {e}"))
    con.commit()
    con.close()
    return {"restored": restored, "failed": failed}


def purge_quarantine(relpaths: list[str] | None = None,
                     quarantine: Path = QUARANTINE, db_path: Path = DB_PATH) -> dict:
    """PERMANENTLY delete files from the trash (irreversible — removes them from disk, i.e. the
    box for the shared library). relpaths=None empties the whole trash. Prunes emptied folders.
    Returns {purged, failed:[(relpath,err)]}."""
    qroot = Path(quarantine)
    if not qroot.exists():
        return {"purged": 0, "failed": []}
    qroot_r = qroot.resolve()                        # _contained returns resolved paths; match it
    con = connect(db_path)
    if relpaths is None:
        relpaths = [d["relpath"] for d in list_quarantine(qroot)]
    purged = 0
    failed: list[tuple[str, str]] = []
    for rel in relpaths:
        p = _contained(qroot, rel)                  # never unlink outside the trash root
        if p is None:
            failed.append((rel, "unsafe path"))
            continue
        try:
            if p.exists():
                p.unlink()
                con.execute("DELETE FROM quarantine WHERE trash_rel=?", (rel,))
                purged += 1
                d = p.parent                        # tidy up now-empty artist/bucket folders
                while d != qroot_r and d.exists() and not any(d.iterdir()):
                    d.rmdir()
                    d = d.parent
        except Exception as e:
            failed.append((rel, f"{type(e).__name__}: {e}"))
    con.commit()
    con.close()
    return {"purged": purged, "failed": failed}


def quarantine_tracks(quarantine: Path = QUARANTINE) -> list[Track]:
    """Trashed files as playable Track objects (path = the trash location, bucket = 'trash'), so the
    app can show + preview them in a TRASH folder view before restoring or deleting. Reads tags for
    artist/title; falls back to the preserved <artist>/<file> path structure."""
    out: list[Track] = []
    for d in list_quarantine(quarantine):
        p = Path(d["abspath"])
        try:
            artist, title, album, duration, bpm = read_tags(p)
        except Exception:
            artist, title, album, duration, bpm = "", "", "", 0.0, None
        out.append(Track(path=d["abspath"], bucket="trash",
                         artist=artist or d["artist"] or p.parent.name, title=title or p.stem,
                         album=album or "", ext=p.suffix.lower(), size=d["size"],
                         mtime=d["mtime"], duration=duration or 0.0, bpm=bpm, key=None))
    return out


def _copy_into_crate(src: Path, dest: Path) -> None:
    """Copy `src` to `dest`, but skip ONLY when `dest` is already byte-identical. The old guard
    skipped on equal SIZE alone, so re-exporting a crate whose `dest` name now maps to a DIFFERENT
    same-size source left the stale audio in place. Compares content when sizes match; otherwise
    writes via a temp file + atomic replace so a crash never leaves a half-written track."""
    if dest.exists() and dest.stat().st_size == src.stat().st_size \
            and filecmp.cmp(str(src), str(dest), shallow=False):
        return
    tmp = dest.with_name(dest.name + ".part")
    shutil.copy2(str(src), str(tmp))
    os.replace(str(tmp), str(dest))


def export(track_paths: list[str], crate_name: str,
           export_root: Path = DEFAULT_EXPORT_ROOT,
           db_path: Path = DB_PATH, copy: bool = True, progress=None) -> dict:
    """Copy the selected tracks to <export_root>\\<crate_name>\\ and write <crate_name>.m3u8
    referencing the LOCAL copies (so you DJ off local files, never the network drive).

    Returns {dest, m3u8, copied, missing:[...], targets:[...]}.
    """
    con = connect(db_path)
    crate_dir = Path(export_root) / _sanitize(crate_name)
    crate_dir.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U"]
    copied = 0
    missing: list[str] = []
    targets: list[str] = []
    used: set[str] = set()

    def _unique(name: str) -> str:
        # de-dup basenames so two tracks called "Intro.flac" don't overwrite each other.
        if name not in used:
            used.add(name)
            return name
        stem, suf = Path(name).stem, Path(name).suffix
        i = 2
        while f"{stem} ({i}){suf}" in used:
            i += 1
        out = f"{stem} ({i}){suf}"
        used.add(out)
        return out

    for i, src_path in enumerate(track_paths):
        row = con.execute("SELECT * FROM tracks WHERE path=?", (src_path,)).fetchone()
        src = Path(src_path)
        if not src.exists():
            missing.append(src_path)
            continue
        artist = row["artist"] if row else src.parent.name
        title = row["title"] if row else src.stem
        dur = int(row["duration"]) if (row and row["duration"]) else -1
        if copy:
            dest = crate_dir / _unique(src.name)
            _copy_into_crate(src, dest)
            target = dest
        else:
            target = src
        lines.append(f"#EXTINF:{dur},{artist} - {title}")
        lines.append(str(target))
        targets.append(str(target))
        copied += 1
        if progress:
            progress(i + 1, src.name)
    con.close()
    m3u8 = crate_dir.with_name(f"{_sanitize(crate_name)}.m3u8")
    m3u8.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"dest": str(crate_dir), "m3u8": str(m3u8), "copied": copied,
            "missing": missing, "targets": targets}


# --- rekordbox XML export (carries the DJ prep: BPM/key/rating/colour/tags/cues) -------------
# Camelot -> musical key for rekordbox's Tonality field (minor = "Xm", major = "X").
CAMELOT_TO_KEY = {
    "1A": "Abm", "1B": "B", "2A": "Ebm", "2B": "F#", "3A": "Bbm", "3B": "Db",
    "4A": "Fm", "4B": "Ab", "5A": "Cm", "5B": "Eb", "6A": "Gm", "6B": "Bb",
    "7A": "Dm", "7B": "F", "8A": "Am", "8B": "C", "9A": "Em", "9B": "G",
    "10A": "Bm", "10B": "D", "11A": "F#m", "11B": "A", "12A": "C#m", "12B": "E",
}
# our colour names -> rekordbox track-colour hex codes
REKORDBOX_COLOUR = {
    "pink": "0xFF007F", "red": "0xFF0000", "orange": "0xFFA500", "yellow": "0xFFFF00",
    "green": "0x00FF00", "aqua": "0x25FDE9", "blue": "0x0000FF", "purple": "0x660099",
}


def _rb_location(path: str) -> str:
    """rekordbox Location URI: file://localhost/ + URL-encoded absolute path, forward slashes.
    Does NOT resolve mapped drives (that would expand Z: to a malformed UNC); export with copy=True
    so Location points at the clean local C: copies you actually DJ from."""
    import urllib.parse
    p = str(path).replace("\\", "/").lstrip("/")          # normalize; drop any leading slashes
    return "file://localhost/" + urllib.parse.quote(p, safe="/:")


def export_rekordbox_xml(track_paths: list[str], crate_name: str,
                         export_root: Path = DEFAULT_EXPORT_ROOT,
                         db_path: Path = DB_PATH, copy: bool = True, progress=None,
                         include_analysis: bool = False) -> dict:
    """Write a rekordbox <collection.xml> for the crate, carrying the DJ prep rekordbox can't
    regenerate: track colour, rating, My-Tags + comment (folded into Comments), and hot/memory
    cues (POSITION_MARK, absolute timestamps). Import via Preferences ▸ Advanced ▸ Database ▸
    rekordbox xml, then drag the playlist out of the 'rekordbox xml' tree.

    By default BPM/beatgrid/key are NOT written — our single-number librosa BPM has no downbeat
    phase, so writing it forces rekordbox onto a wrong grid ("off grid"). Leaving it out lets
    rekordbox analyze the grid + key itself (which it does well) while our prep rides along.
    Set include_analysis=True to also write AverageBpm/TEMPO/Tonality (only if you trust them).

    If copy=True the audio is copied next to the xml and Location points at the copies; else it
    points at the original files. Returns {dest, xml, copied, missing, targets}.
    """
    import xml.etree.ElementTree as ET
    import xml.dom.minidom as minidom

    con = connect(db_path)
    crate_dir = Path(export_root) / _sanitize(crate_name)
    crate_dir.mkdir(parents=True, exist_ok=True)

    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="Crate", Version="1.0", Company="crate")
    collection = ET.SubElement(root, "COLLECTION")
    playlists = ET.SubElement(root, "PLAYLISTS")
    proot = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="1")
    pnode = ET.SubElement(proot, "NODE", Name=_sanitize(crate_name), Type="1", KeyType="0")

    missing: list[str] = []
    targets: list[str] = []
    used: set[str] = set()
    copied = 0
    entries = 0

    def _unique(name: str) -> str:
        if name not in used:
            used.add(name)
            return name
        stem, suf = Path(name).stem, Path(name).suffix
        i = 2
        while f"{stem} ({i}){suf}" in used:
            i += 1
        out = f"{stem} ({i}){suf}"
        used.add(out)
        return out

    for i, src_path in enumerate(track_paths):
        row = con.execute("SELECT * FROM tracks WHERE path=?", (src_path,)).fetchone()
        src = Path(src_path)
        if not src.exists():
            missing.append(src_path)
            continue
        if copy:
            dest = crate_dir / _unique(src.name)
            _copy_into_crate(src, dest)
            target = dest
        else:
            target = src
        targets.append(str(target))
        copied += 1
        entries += 1
        tid = str(entries)

        artist = (row["artist"] if row else src.parent.name) or ""
        title = (row["title"] if row else src.stem) or ""
        album = (row["album"] if row else "") or ""
        dur = int(row["duration"]) if (row and row["duration"]) else 0
        bpm = row["bpm"] if (row and row["bpm"]) else None
        key = (row["key"] if row else None) or None
        rating = int(row["rating"]) if (row and row["rating"]) else 0
        colour = (row["color"] if row else None) or None
        comment = (row["comment"] if row else None) or ""

        # fold My-Tags into the Comments field (rekordbox xml has no native tag slot)
        tagmap = get_track_tags(src_path, db_path=db_path)
        flat_tags = [v for vals in tagmap.values() for v in vals]
        combined = ", ".join(flat_tags)
        if comment:
            combined = f"{combined}  |  {comment}" if combined else comment

        attrs = {
            "TrackID": tid, "Name": title, "Artist": artist, "Album": album,
            "Kind": f"{src.suffix.lstrip('.').upper()} File",
            "TotalTime": str(dur), "Location": _rb_location(str(target)),
        }
        if include_analysis and bpm:    # off by default — let rekordbox compute the real grid
            attrs["AverageBpm"] = f"{float(bpm):.2f}"
        if include_analysis and key:
            attrs["Tonality"] = CAMELOT_TO_KEY.get(str(key).upper(), str(key))
        if rating:
            attrs["Rating"] = str(min(5, max(0, rating)) * 51)   # rekordbox: 0/51/102/153/204/255
        if colour and colour in REKORDBOX_COLOUR:
            attrs["Colour"] = REKORDBOX_COLOUR[colour]
        if combined:
            attrs["Comments"] = combined
        track_el = ET.SubElement(collection, "TRACK", **attrs)

        if include_analysis and bpm:    # a single starting anchor (we have no real beatgrid)
            ET.SubElement(track_el, "TEMPO", Inizio="0.000", Bpm=f"{float(bpm):.2f}",
                          Metro="4/4", Battito="1")
        for c in get_cues(src_path, db_path=db_path):
            start = f"{(c['position_ms'] or 0) / 1000.0:.3f}"
            if c["kind"] == "hot":
                try:
                    num = max(0, int(str(c["idx"])) - 1)
                except (TypeError, ValueError):
                    num = 0
            else:
                num = -1               # memory cue
            ET.SubElement(track_el, "POSITION_MARK", Name=(c.get("name") or ""),
                          Type="0", Start=start, Num=str(num))

        ET.SubElement(pnode, "TRACK", Key=tid)
        if progress:
            progress(i + 1, src.name)

    con.close()
    collection.set("Entries", str(entries))
    pnode.set("Entries", str(entries))

    xml_path = crate_dir.with_name(f"{_sanitize(crate_name)}.xml")
    pretty = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
    xml_path.write_text(pretty, encoding="utf-8")
    return {"dest": str(crate_dir), "xml": str(xml_path), "copied": copied,
            "missing": missing, "targets": targets}


# --- saved crates (each crate is a folder under the crates root) -------------
def save_crate(name: str, track_paths: list[str], crates_root: Path | None = None,
               db_path: Path = DB_PATH, progress=None) -> dict:
    """Persist a crate as a folder: copy files + write <name>.m3u8 AND a rekordbox <name>.xml
    (carrying BPM/key/rating/colour/tags/cues) AND a manifest of the ORIGINAL library paths so the
    app can reopen + map it later. Saving a crate makes it instantly rekordbox-ready — no separate
    export step. Returns export()'s dict plus {'manifest', 'xml'}. The folder IS the saved crate.
    """
    root = Path(crates_root) if crates_root is not None else get_crates_root()
    res = export(track_paths, name, export_root=root, db_path=db_path, copy=True, progress=progress)
    crate_dir = Path(res["dest"])
    # reconcile: drop any leftover audio copies from a previous save of this crate that aren't
    # part of the current track set (so re-saving a trimmed crate doesn't accumulate orphans).
    keep = {Path(t).name for t in res["targets"]}
    for f in crate_dir.iterdir():
        if (f.is_file() and f.suffix.lower() in AUDIO_EXTS and f.name not in keep):
            try:
                f.unlink()
            except OSError:
                pass
    # rekordbox xml next to the m3u8 — files are already copied, so this just writes the xml
    rb = export_rekordbox_xml(track_paths, name, export_root=root, db_path=db_path, copy=True)
    res["xml"] = rb["xml"]
    man = crate_dir / CRATE_MANIFEST
    man.write_text("\n".join(track_paths) + "\n", encoding="utf-8")
    res["manifest"] = str(man)
    return res


def list_crates(crates_root: Path | None = None) -> list[tuple[str, int, float]]:
    """[(name, track_count, mtime)] for every saved-crate folder (has a manifest), newest first."""
    root = Path(crates_root) if crates_root is not None else get_crates_root()
    out: list[tuple[str, int, float]] = []
    if not root.exists():
        return out
    for d in root.iterdir():
        man = d / CRATE_MANIFEST
        if d.is_dir() and man.exists():
            try:
                n = len([ln for ln in man.read_text(encoding="utf-8").splitlines() if ln.strip()])
            except Exception:
                n = 0
            out.append((d.name, n, man.stat().st_mtime))
    out.sort(key=lambda x: x[2], reverse=True)
    return out


def read_crate(name: str, crates_root: Path | None = None,
               db_path: Path = DB_PATH) -> list[Track]:
    """Reopen a saved crate: resolve its manifest paths back to Tracks (skip any now-missing)."""
    root = Path(crates_root) if crates_root is not None else get_crates_root()
    man = root / _sanitize(name) / CRATE_MANIFEST
    if not man.exists():
        return []
    con = connect(db_path)
    tracks: list[Track] = []
    for line in man.read_text(encoding="utf-8").splitlines():
        p = line.strip()
        if not p:
            continue
        r = con.execute("SELECT * FROM tracks WHERE path=?", (p,)).fetchone()
        if r:
            tracks.append(_row_to_track(r))
    con.close()
    return tracks


def delete_crate(name: str, crates_root: Path | None = None) -> bool:
    root = Path(crates_root) if crates_root is not None else get_crates_root()
    d = root / _sanitize(name)
    if d.is_dir():
        shutil.rmtree(d)
        (root / f"{_sanitize(name)}.m3u8").unlink(missing_ok=True)  # also drop the sibling playlist
        return True
    return False


def rename_crate(old: str, new: str, crates_root: Path | None = None) -> bool:
    root = Path(crates_root) if crates_root is not None else get_crates_root()
    src, dst = root / _sanitize(old), root / _sanitize(new)
    if src.is_dir() and not dst.exists():
        src.rename(dst)
        old_m3u8 = root / f"{_sanitize(old)}.m3u8"
        if old_m3u8.exists():
            old_m3u8.rename(root / f"{_sanitize(new)}.m3u8")
        return True
    return False


# --- cli (for testing without the GUI) --------------------------------------
def _main(argv=None):
    ap = argparse.ArgumentParser(description="Crate library core")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("sync")
    sp = sub.add_parser("search")
    sp.add_argument("query", nargs="?", default="")
    sp.add_argument("--bucket")
    ep = sub.add_parser("export")
    ep.add_argument("name")
    ep.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)

    if args.cmd == "index":
        print(index())
    elif args.cmd == "sync":
        print(f"features merged into {sync_features()} tracks")
    elif args.cmd == "search":
        for t in search(args.query, bucket=args.bucket, limit=50):
            print(f"[{t.bucket}] {t.artist} - {t.title}  ({t.ext})")
    elif args.cmd == "export":
        print(export(args.paths, args.name))


if __name__ == "__main__":
    _main()
