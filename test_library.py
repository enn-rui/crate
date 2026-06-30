"""Unit tests for library.py — index/search/export against a temp fixture library.

Uses dummy files (no real audio), so tags come from the <Artist>\\<Title> path fallback
and no network/mutagen-parsing is required. Run: python -m pytest test_library.py -q
"""
from pathlib import Path

import pytest

import library


def _make_lib(tmp: Path) -> Path:
    """Build a fake library matching the real bucket layout: <root>/<bucket-relpath>/<Artist>/<Title>.<ext>."""
    rel = {lbl: r for lbl, r in library.BUCKETS}   # dj -> music/dj, personal -> music/personal, mp3 -> music-mp3
    layout = {
        "dj": [("Nia Archives", "Sober Feels", ".flac"),
               ("Machinedrum", "GBYE", ".flac")],
        "personal": [("Karol G", "Provenza (Sistek Remix)", ".flac")],
        "mp3": [("Cobrah", "IDFKA", ".mp3")],
    }
    for bucket, tracks in layout.items():
        for artist, title, ext in tracks:
            f = tmp / rel[bucket] / artist / f"{title}{ext}"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"\x00" * 2048)  # non-zero size; not real audio
    return tmp


def test_index_counts(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    res = library.index(root=root, db_path=db)
    assert res["total"] == 4
    assert res["added"] == 4
    # re-index is idempotent: everything skipped, nothing added
    res2 = library.index(root=root, db_path=db)
    assert res2["added"] == 0 and res2["skipped"] == 4 and res2["total"] == 4


def test_reindex_self_heals_relabeled_bucket(tmp_path):
    # same files, but a scan root gets a new label: skipped (unchanged) files must adopt the new bucket
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    mp3_src = [("oldlabel", str(root / "music-mp3"))]
    library.index(db_path=db, sources=mp3_src)
    assert library.search("", bucket="oldlabel", db_path=db)
    res = library.index(db_path=db, sources=[("mp3", str(root / "music-mp3"))])
    assert res["skipped"] == 1 and res["added"] == 0           # unchanged file, fast path
    assert not library.search("", bucket="oldlabel", db_path=db)   # stale label gone
    assert len(library.search("", bucket="mp3", db_path=db)) == 1  # adopted new label


def test_index_prunes_deleted(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    (root / "music-mp3" / "Cobrah" / "IDFKA.mp3").unlink()
    res = library.index(root=root, db_path=db)
    assert res["removed"] == 1 and res["total"] == 3


def test_path_fallback_tags(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    hits = library.search("Nia Archives", db_path=db)
    assert len(hits) == 1
    assert hits[0].artist == "Nia Archives" and hits[0].title == "Sober Feels"
    assert hits[0].bucket == "dj"


def test_search_filters(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    assert len(library.search("", db_path=db)) == 4
    assert len(library.search("", bucket="dj", db_path=db)) == 2     # Nia Archives + Machinedrum
    assert len(library.search("", bucket="personal", db_path=db)) == 1
    assert len(library.search("Provenza", db_path=db)) == 1  # title match
    assert len(library.search("nonexistent", db_path=db)) == 0


def test_export_copies_and_writes_m3u8(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    paths = [t.path for t in library.search("", bucket="dj", db_path=db)]
    out = tmp_path / "out"
    res = library.export(paths, "Test Set", export_root=out, db_path=db)
    crate_dir = Path(res["dest"])
    assert crate_dir.is_dir()
    copied = list(crate_dir.glob("*.flac"))
    assert len(copied) == 2 and res["copied"] == 2 and res["missing"] == []
    m3u8 = Path(res["m3u8"])
    text = m3u8.read_text(encoding="utf-8")
    assert text.startswith("#EXTM3U")
    assert "#EXTINF" in text
    # m3u8 references the LOCAL copies, not the source library
    for c in copied:
        assert str(c) in text


def test_rekordbox_xml_carries_prep(tmp_path):
    import xml.etree.ElementTree as ET
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    paths = [t.path for t in library.search("", bucket="dj", db_path=db)]
    p0 = paths[0]
    # attach the full DJ prep onto one track
    con = library.connect(db)
    con.execute("UPDATE tracks SET bpm=?, key=? WHERE path=?", (128.0, "8A", p0))
    con.commit(); con.close()
    library.set_color(p0, "aqua", db_path=db)
    library.set_rating(p0, 4, db_path=db)
    library.set_comment(p0, "peak time", db_path=db)
    library.add_track_tag(p0, "genre", "techno", db_path=db)
    library.add_cue(p0, "memory", "1", 30500, db_path=db)
    library.add_cue(p0, "hot", "1", 61000, db_path=db)

    out = tmp_path / "out"
    res = library.export_rekordbox_xml(paths, "RB Set", export_root=out, db_path=db, copy=True)
    assert Path(res["xml"]).is_file() and res["copied"] == 2
    tr = ET.parse(res["xml"]).getroot().findall(".//COLLECTION/TRACK")[0]
    # by default BPM/key are OMITTED so rekordbox analyzes the grid itself (no "off grid")
    assert tr.get("AverageBpm") is None and tr.get("Tonality") is None
    assert tr.find("TEMPO") is None
    # but the user's prep rekordbox can't regenerate IS carried:
    assert tr.get("Rating") == "204"                     # 4 stars * 51
    assert tr.get("Colour") == "0x25FDE9"                # aqua
    assert "techno" in tr.get("Comments") and "peak time" in tr.get("Comments")
    assert tr.get("Location").startswith("file://localhost/")
    marks = tr.findall("POSITION_MARK")
    assert len(marks) == 2
    assert any(m.get("Num") == "-1" and m.get("Start") == "30.500" for m in marks)   # memory
    assert any(m.get("Num") == "0" and m.get("Start") == "61.000" for m in marks)    # hot A
    # playlist node references both tracks
    pn = ET.parse(res["xml"]).getroot().find(".//PLAYLISTS/NODE/NODE")
    assert pn.get("Entries") == "2" and [t.get("Key") for t in pn.findall("TRACK")] == ["1", "2"]
    # opt-in: include_analysis=True does write BPM/key/TEMPO for anyone who wants the hint
    res2 = library.export_rekordbox_xml(paths, "RB Set2", export_root=out, db_path=db,
                                        copy=True, include_analysis=True)
    tr2 = ET.parse(res2["xml"]).getroot().findall(".//COLLECTION/TRACK")[0]
    assert tr2.get("AverageBpm") == "128.00" and tr2.get("Tonality") == "Am" and tr2.find("TEMPO") is not None


def test_camelot_neighbors():
    n = library.camelot_neighbors("8A")
    assert set(n) == {"8A", "8B", "7A", "9A"}
    assert n["8A"] == "same key" and n["8B"] == "relative major/minor"
    assert library.camelot_neighbors("12A") == {"12A": "same key", "12B": "relative major/minor",
                                                "11A": "-1 (energy down)", "1A": "+1 (energy up)"}
    assert library.camelot_neighbors("?") == {}


def test_harmonic_matches(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    paths = [t.path for t in library.search("", db_path=db)]
    con = library.connect(db)
    con.execute("UPDATE tracks SET key='8A', bpm=128 WHERE path=?", (paths[0],))  # seed
    con.execute("UPDATE tracks SET key='9A', bpm=130 WHERE path=?", (paths[1],))  # compatible
    con.execute("UPDATE tracks SET key='3B', bpm=128 WHERE path=?", (paths[2],))  # wrong key
    con.execute("UPDATE tracks SET key='8A', bpm=200 WHERE path=?", (paths[3],))  # key ok, bpm off
    con.commit()
    con.close()
    seed = next(t for t in library.search("", db_path=db) if t.path == paths[0])
    mp = [m.path for m in library.harmonic_matches(seed, db_path=db)]
    assert paths[1] in mp            # 9A@130 mixes
    assert paths[2] not in mp        # 3B incompatible key
    assert paths[3] not in mp        # 8A but BPM out of range
    assert paths[0] not in mp        # never the seed itself


def test_set_rating(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    p = library.search("", db_path=db)[0].path
    library.set_rating(p, 4, db_path=db)
    assert next(t for t in library.search("", db_path=db) if t.path == p).rating == 4
    library.set_rating(p, 0, db_path=db)  # 0 clears
    assert next(t for t in library.search("", db_path=db) if t.path == p).rating is None


def test_track_tags_crud_and_values(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    paths = [t.path for t in library.search("", db_path=db)]
    p = paths[0]
    other = paths[1]
    library.set_track_tags(p, "mood", ["dark", " rolling ", "", "dark"], db_path=db)
    assert library.get_track_tags(p, db_path=db) == {"mood": ["dark", "rolling"]}
    library.set_track_tags(p, "mood", ["bright"], db_path=db)
    assert library.get_track_tags(p, db_path=db) == {"mood": ["bright"]}
    library.add_track_tag(p, "genre", "jungle", db_path=db)
    library.add_track_tag(p, "genre", "breaks", db_path=db)
    library.remove_track_tag(p, "genre", "jungle", db_path=db)
    assert library.get_track_tags(p, db_path=db) == {"genre": ["breaks"], "mood": ["bright"]}
    library.add_track_tag(other, "genre", "ambient", db_path=db)
    library.add_track_tag(other, "genre", "breaks", db_path=db)
    assert library.all_tag_values("genre", db_path=db) == ["ambient", "breaks"]


def test_color_comment_roundtrip_and_clear(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    p = library.search("", db_path=db)[0].path
    library.set_color(p, "blue", db_path=db)
    library.set_comment(p, "opener", db_path=db)
    t = next(t for t in library.search("", db_path=db) if t.path == p)
    assert t.color == "blue" and t.comment == "opener"
    con = library.connect(db)
    row = con.execute("SELECT color, comment FROM tracks WHERE path=?", (p,)).fetchone()
    con.close()
    assert row["color"] == "blue" and row["comment"] == "opener"
    library.set_color(p, None, db_path=db)
    library.set_comment(p, None, db_path=db)
    t = next(t for t in library.search("", db_path=db) if t.path == p)
    assert t.color is None and t.comment is None


def test_dj_metadata_survives_reindex(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    p = library.search("", db_path=db)[0].path
    library.set_color(p, "green", db_path=db)
    library.set_comment(p, "blend with drums", db_path=db)
    library.set_track_tags(p, "situation", ["warmup"], db_path=db)
    cue_id = library.add_cue(p, "hot", "A", 12000, color="green", name="drop", db_path=db)
    library.index(root=root, db_path=db)
    t = next(t for t in library.search("", db_path=db) if t.path == p)
    assert t.color == "green" and t.comment == "blend with drums"
    assert library.get_track_tags(p, db_path=db) == {"situation": ["warmup"]}
    assert library.get_cues(p, db_path=db) == [{
        "id": cue_id,
        "kind": "hot",
        "idx": "A",
        "position_ms": 12000,
        "color": "green",
        "name": "drop",
    }]


def test_cues_crud(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    p = library.search("", db_path=db)[0].path
    late = library.add_cue(p, "memory", "2", 40000, db_path=db)
    early = library.add_cue(p, "hot", "A", 10000, color="red", name="intro", db_path=db)
    middle = library.add_cue(p, "memory", "1", 25000, db_path=db)
    assert [c["id"] for c in library.get_cues(p, db_path=db)] == [early, middle, late]
    library.delete_cue(middle, db_path=db)
    assert [c["id"] for c in library.get_cues(p, db_path=db)] == [early, late]
    library.clear_cues(p, db_path=db)
    assert library.get_cues(p, db_path=db) == []


def test_delete_quarantines_and_unindexes(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    victim = root / "music-mp3" / "Cobrah" / "IDFKA.mp3"
    quar = tmp_path / "trash"
    res = library.delete_tracks([str(victim)], db_path=db, quarantine=quar, lib_root=root)
    assert res["moved"] == 1 and res["failed"] == []
    # file is gone from the library but preserved in quarantine (reversible)
    assert not victim.exists()
    assert (quar / "music-mp3" / "Cobrah" / "IDFKA.mp3").exists()
    # and dropped from the index
    assert library.search("Cobrah", db_path=db) == []
    assert len(library.search("", db_path=db)) == 3


def test_trash_list_restore_purge(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    quar = tmp_path / "trash"
    victim = root / "music-mp3" / "Cobrah" / "IDFKA.mp3"
    library.delete_tracks([str(victim)], db_path=db, quarantine=quar, lib_root=root)

    # list shows the trashed file with parsed bucket/artist
    listed = library.list_quarantine(quarantine=quar)
    assert len(listed) == 1
    assert listed[0]["relpath"] == "music-mp3/Cobrah/IDFKA.mp3"
    assert listed[0]["artist"] == "Cobrah" and listed[0]["bucket"] == "music-mp3"

    # restore puts it back on disk AND re-indexes it
    res = library.restore_tracks(["music-mp3/Cobrah/IDFKA.mp3"], quarantine=quar,
                                 lib_root=root, db_path=db)
    assert res["restored"] == 1 and res["failed"] == []
    assert victim.exists() and library.list_quarantine(quarantine=quar) == []
    assert len(library.search("Cobrah", db_path=db)) == 1

    # delete again, then purge for good — file is gone from disk and the trash is empty
    library.delete_tracks([str(victim)], db_path=db, quarantine=quar, lib_root=root)
    pres = library.purge_quarantine(quarantine=quar)        # None => empty all
    assert pres["purged"] == 1
    assert library.list_quarantine(quarantine=quar) == [] and not victim.exists()

    # path-escape guard: a relpath with .. is refused, nothing deleted outside the trash
    guard = library.purge_quarantine(["../../secret.flac"], quarantine=quar)
    assert guard["purged"] == 0 and guard["failed"]


def test_purge_refuses_absolute_path(tmp_path):
    # an ABSOLUTE relpath must not escape the trash (qroot / "/abs" == "/abs" in pathlib)
    quar = tmp_path / "trash"; quar.mkdir()
    victim = tmp_path / "keep.flac"; victim.write_bytes(b"\x00" * 16)
    res = library.purge_quarantine([str(victim)], quarantine=quar, db_path=tmp_path / "t.db")
    assert res["purged"] == 0 and res["failed"]
    assert victim.exists()                                  # the outside file is untouched


def test_restore_returns_file_to_original_outside_lib_root(tmp_path):
    # delete a file from an added scan root OUTSIDE lib_root, then restore — it must go back to its
    # exact original location (the manifest), not lib_root/<basename>.
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    quar = tmp_path / "trash"
    outside = tmp_path / "elsewhere" / "Gem.flac"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"\x01" * 4096)
    library.delete_tracks([str(outside)], db_path=db, quarantine=quar, lib_root=root)
    assert not outside.exists() and (quar / "Gem.flac").exists()
    res = library.restore_tracks(["Gem.flac"], quarantine=quar, lib_root=root, db_path=db)
    assert res["restored"] == 1 and res["failed"] == []
    assert outside.exists()                                 # back where it came from, not under root
    assert not (root / "Gem.flac").exists()


def test_export_overwrites_stale_same_size_copy(tmp_path):
    # re-exporting a crate name where a basename now maps to a DIFFERENT same-size source must
    # refresh the copied audio, not trust the existing same-size file.
    db = tmp_path / "t.db"
    a = tmp_path / "A" / "Intro.flac"; a.parent.mkdir(parents=True); a.write_bytes(b"A" * 2048)
    b = tmp_path / "B" / "Intro.flac"; b.parent.mkdir(parents=True); b.write_bytes(b"B" * 2048)
    out = tmp_path / "out"
    library.export([str(a)], "Set", export_root=out, db_path=db, copy=True)
    dest = out / "Set" / "Intro.flac"
    assert dest.read_bytes() == b"A" * 2048
    library.export([str(b)], "Set", export_root=out, db_path=db, copy=True)
    assert dest.read_bytes() == b"B" * 2048                 # stale A audio was replaced


def test_export_reports_missing(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    res = library.export([str(root / "music" / "Ghost" / "missing.flac")],
                         "Broken", export_root=tmp_path / "out2", db_path=db)
    assert res["copied"] == 0 and len(res["missing"]) == 1


def test_save_list_read_crate_roundtrip(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    crates = tmp_path / "crates"
    paths = [t.path for t in library.search("", bucket="dj", db_path=db)]  # 2 tracks
    res = library.save_crate("Peak Set", paths, crates_root=crates, db_path=db)
    # folder persisted with copies + m3u8 + manifest + rekordbox xml
    crate_dir = Path(res["dest"])
    assert crate_dir.is_dir() and Path(res["manifest"]).exists()
    assert len(list(crate_dir.glob("*.flac"))) == 2
    assert Path(res["xml"]).exists() and Path(res["xml"]).suffix == ".xml"   # saving = rekordbox-ready
    # list_crates surfaces it with the right count
    crs = library.list_crates(crates_root=crates)
    assert len(crs) == 1 and crs[0][0] == "Peak Set" and crs[0][1] == 2
    # read_crate resolves the manifest back to the SAME library tracks (originals, not copies)
    reopened = library.read_crate("Peak Set", crates_root=crates, db_path=db)
    assert sorted(t.path for t in reopened) == sorted(paths)
    # rename + delete
    assert library.rename_crate("Peak Set", "Warmup", crates_root=crates)
    assert {c[0] for c in library.list_crates(crates_root=crates)} == {"Warmup"}
    assert library.delete_crate("Warmup", crates_root=crates)
    assert library.list_crates(crates_root=crates) == []


def test_config_roundtrip_and_index(tmp_path):
    # config persists and drives a multi-root index via sources / config_sources()
    cfg_path = tmp_path / "cfg.json"
    a = _make_lib(tmp_path / "libA")            # has music/dj (2) + music/personal (1) + music-mp3 (1)
    extra = tmp_path / "extra" / "Bonus"
    extra.mkdir(parents=True)
    (extra / "Track.flac").write_bytes(b"\x00" * 2048)
    cfg = {"scan_roots": [{"label": "dj", "path": str(a / "music" / "dj")},
                          {"label": "extra", "path": str(tmp_path / "extra")}],
           "crates_root": str(tmp_path / "crates")}
    library.save_config(cfg, path=cfg_path)
    assert library.load_config(path=cfg_path)["scan_roots"][1]["label"] == "extra"
    db = tmp_path / "t.db"
    sources = [(r["label"], r["path"]) for r in cfg["scan_roots"]]
    res = library.index(db_path=db, sources=sources)
    assert res["total"] == 3                    # 2 from music/dj + 1 from extra/
    assert len(library.search("", bucket="extra", db_path=db)) == 1


def test_default_config_when_missing(tmp_path):
    cfg = library.load_config(path=tmp_path / "nope.json")
    assert [r["label"] for r in cfg["scan_roots"]] == [lbl for lbl, _ in library.BUCKETS]


def test_load_config_preserves_analysis_remote(tmp_path):
    # load_config is an allowlist; analysis_remote must survive a file round-trip or the
    # ANALYZE button never sees it (and an in-app save would wipe it).
    cfg_path = tmp_path / "c.json"
    library.save_config({"crates_root": str(tmp_path),
                         "analysis_remote": {"ssh": "user@host", "root": "/music"}}, path=cfg_path)
    assert library.load_config(path=cfg_path)["analysis_remote"]["ssh"] == "user@host"


def test_index_skips_prune_when_root_unreachable(tmp_path):
    # H1: if a scan root disappears, re-index must NOT wipe the index (ratings live only here)
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    p = library.search("", db_path=db)[0].path
    library.set_rating(p, 5, db_path=db)
    # now point at a root that doesn't exist (simulates Z: disconnected)
    res = library.index(db_path=db, sources=[("music", tmp_path / "GONE")])
    assert res["pruned"] is False
    assert res["removed"] == 0
    assert res["total"] == 4                       # nothing pruned
    assert str(tmp_path / "GONE") in res["missing_roots"]
    assert next(t for t in library.search("", db_path=db) if t.path == p).rating == 5  # rating kept


def test_export_dedupes_colliding_filenames(tmp_path):
    # M2: two tracks with the same basename must not overwrite each other
    a = tmp_path / "lib" / "music" / "Artist A" / "Intro.flac"
    b = tmp_path / "lib" / "music" / "Artist B" / "Intro.flac"
    for f, n in ((a, 1111), (b, 2222)):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"\x00" * n)
    db = tmp_path / "t.db"
    library.index(root=tmp_path / "lib", db_path=db)
    out = tmp_path / "out"
    res = library.export([str(a), str(b)], "Dupes", export_root=out, db_path=db)
    assert res["copied"] == 2 and len(set(res["targets"])) == 2  # distinct destinations
    assert {Path(t).stat().st_size for t in res["targets"]} == {1111, 2222}  # both preserved


def test_save_crate_reconciles_orphans(tmp_path):
    # M3: re-saving a trimmed crate removes the dropped track's leftover copy
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    crates = tmp_path / "crates"
    paths = [t.path for t in library.search("", db_path=db)]
    library.save_crate("Set", paths, crates_root=crates, db_path=db)
    n_before = len(list((crates / "Set").glob("*.*")))
    library.save_crate("Set", paths[:1], crates_root=crates, db_path=db)  # trim to 1 track
    audio_after = [f for f in (crates / "Set").iterdir()
                   if f.suffix.lower() in library.AUDIO_EXTS]
    assert len(audio_after) == 1 and n_before > 1
    assert library.read_crate("Set", crates_root=crates, db_path=db)[0].path == paths[0]


# --- CLAP similarity + mixability -------------------------------------------
def _write_vectors(root: Path, vecs: dict[str, list[float]]) -> Path:
    """Write a music_vectors.sqlite (embed_clap.py schema) under <root>/.crate/.
    `vecs` maps relpath (forward-slash, relative to root) -> raw float list."""
    import sqlite3, struct, time
    d = root / ".crate"
    d.mkdir(parents=True, exist_ok=True)
    db = d / "music_vectors.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("""CREATE TABLE IF NOT EXISTS vectors (
        relpath TEXT PRIMARY KEY, mtime REAL, dim INTEGER, vec BLOB, added_at REAL)""")
    for rel, v in vecs.items():
        blob = struct.pack(f"<{len(v)}f", *v)
        con.execute("INSERT OR REPLACE INTO vectors VALUES(?,?,?,?,?)",
                    (rel, 0.0, len(v), blob, time.time()))
    con.commit()
    con.close()
    return db


def _write_clusters(root: Path, labels: dict[str, int]) -> Path:
    """Write a clusters.sqlite sidecar under <root>/.crate/."""
    import sqlite3
    d = root / ".crate"
    d.mkdir(parents=True, exist_ok=True)
    db = d / "clusters.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE clusters (relpath TEXT PRIMARY KEY, cluster_id INTEGER)")
    for rel, cluster_id in labels.items():
        con.execute("INSERT INTO clusters VALUES(?, ?)", (rel, cluster_id))
    con.commit()
    con.close()
    return db


def test_load_clusters_roundtrips(tmp_path):
    root = tmp_path / "lib"
    cp = _write_clusters(root, {
        "music/dj/A/One.flac": 4,
        "music/dj/B/Two.flac": -1,
        "music/personal/C/Three.flac": 12,
    })
    library.clear_vector_cache()
    clusters = library.load_clusters(clusters_path=cp, lib_root=root, force=True)
    assert clusters == {
        str(root / "music/dj/A/One.flac"): 4,
        str(root / "music/dj/B/Two.flac"): -1,
        str(root / "music/personal/C/Three.flac"): 12,
    }
    library.clear_vector_cache()


def test_load_clusters_missing_returns_empty(tmp_path):
    library.clear_vector_cache()
    assert library.load_clusters(clusters_path=tmp_path / "missing.sqlite",
                                 lib_root=tmp_path / "lib", force=True) == {}


def test_compat_penalty_identical_and_clashing(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    a, b, c = library.search("", db_path=db)[:3]
    a.bpm, a.key = 128.0, "8A"
    b.bpm, b.key = 128.0, "8A"
    c.bpm, c.key = 150.0, "3B"
    assert abs(library.compat_penalty(a, b) - 0.0) < 1e-9
    assert library.compat_penalty(a, c) > library.compat_penalty(a, b)


def test_compatible_next_and_build_path(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    paths = [t.path for t in library.search("", db_path=db)]
    con = library.connect(db)
    for p in paths:                                   # all mutually compatible (same key + bpm)
        con.execute("UPDATE tracks SET key='8A', bpm=128 WHERE path=?", (p,))
    con.commit(); con.close()
    library.clear_vector_cache()
    seed = next(t for t in library.search("", db_path=db) if t.path == paths[0])

    nxt = library.compatible_next(seed, db_path=db)
    assert nxt and all(0.0 <= sc <= 1.0 for _, sc in nxt)     # scored candidates
    npaths = [t.path for t, _ in nxt]
    assert paths[0] not in npaths                             # never the seed itself
    nxt2 = library.compatible_next(seed, exclude_paths=[paths[1]], db_path=db)
    assert paths[1] not in [t.path for t, _ in nxt2]          # exclusion honored

    path = library.build_path(seed, length=4, db_path=db)
    assert path[0].path == paths[0]                           # starts at the seed
    pp = [t.path for t in path]
    assert len(pp) == len(set(pp))                            # no repeats
    assert 2 <= len(path) <= 4                                # chained, bounded by length


def test_analysis_python_path_from_config(monkeypatch, tmp_path):
    fake = tmp_path / "python.exe"
    fake.write_text("")
    monkeypatch.setattr(library, "load_config", lambda *a, **k: {"analysis_python": str(fake)})
    assert library.analysis_python_path() == fake


def test_run_analysis_requires_env(monkeypatch):
    monkeypatch.setattr(library, "analysis_python_path", lambda: None)
    with pytest.raises(RuntimeError):
        library.run_analysis(root=Path("."))


def test_analysis_remote_config_none_without_ssh(monkeypatch):
    monkeypatch.setattr(library, "load_config", lambda *a, **k: {})
    assert library.analysis_remote_config() is None
    monkeypatch.setattr(library, "load_config", lambda *a, **k: {"analysis_remote": {"python": "x"}})
    assert library.analysis_remote_config() is None  # ssh is required


def test_analysis_remote_config_defaults(monkeypatch):
    monkeypatch.setattr(library, "load_config",
                        lambda *a, **k: {"analysis_remote": {"ssh": "user@host"}})
    rc = library.analysis_remote_config()
    assert rc["ssh"] == "user@host"
    assert rc["script"].endswith("analyze_all.py") and rc["root"] == "/path/to/music"


def test_run_analysis_remote_requires_config(monkeypatch):
    monkeypatch.setattr(library, "analysis_remote_config", lambda: None)
    with pytest.raises(RuntimeError):
        library.run_analysis_remote()


def test_energy_pick_arc():
    class T:                                                  # minimal track stand-in
        def __init__(self, e): self.energy = e
    cur = T(0.5)
    lo, hi = T(0.2), T(0.8)
    cands = [(lo, 0.9), (hi, 0.8)]                            # lo mixes best, hi has more energy
    assert library._energy_pick(cur, cands, "flat") is lo    # best mixability
    assert library._energy_pick(cur, cands, "up") is hi      # rising arc skips the energy drop
    assert library._energy_pick(cur, cands, "down") is lo     # falling arc keeps the lower-energy one


def _write_section_vectors(root: Path, secs: dict[str, tuple[list[float], list[float]]]) -> Path:
    """Write a music_vectors.sqlite WITH vec_intro/vec_outro columns (newer embed_clap.py schema).
    `secs` maps relpath -> (intro_raw, outro_raw)."""
    import sqlite3, struct, time
    d = root / ".crate"; d.mkdir(parents=True, exist_ok=True)
    db = d / "music_vectors.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("""CREATE TABLE IF NOT EXISTS vectors (
        relpath TEXT PRIMARY KEY, mtime REAL, dim INTEGER, vec BLOB,
        vec_intro BLOB, vec_outro BLOB, added_at REAL)""")
    for rel, (iv, ov) in secs.items():
        whole = struct.pack(f"<{len(iv)}f", *iv)
        con.execute("INSERT OR REPLACE INTO vectors VALUES(?,?,?,?,?,?,?)",
                    (rel, 0.0, len(iv), whole,
                     struct.pack(f"<{len(iv)}f", *iv), struct.pack(f"<{len(ov)}f", *ov), time.time()))
    con.commit(); con.close()
    return db


def test_transition_score_directional_and_drives_mixability(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    tracks = library.search("", db_path=db)
    a, b, c = tracks[0], tracks[1], tracks[2]
    rels = {t.path: Path(t.path).relative_to(root).as_posix() for t in (a, b, c)}
    # A's OUTRO points +x; B's INTRO points +x (A->B flows perfectly), C's INTRO points +y (clash).
    vp = _write_section_vectors(root, {
        rels[a.path]: ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),   # (intro, outro) — outro=+x
        rels[b.path]: ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),   # intro=+x  -> A->B great
        rels[c.path]: ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),   # intro=+y  -> A->C poor
    })
    library.clear_vector_cache()
    secs = library.load_section_vectors(vectors_path=vp, lib_root=root, force=True)
    assert len(secs) == 3
    s_ab = library.transition_score(a.path, b.path, secs)
    s_ac = library.transition_score(a.path, c.path, secs)
    assert s_ab > 0.99 and s_ac < 0.01                       # A's outro fits B's intro, not C's
    # directional: B's outro (+y) vs A's intro (+x) is a clash even though A->B was perfect
    assert library.transition_score(b.path, a.path, secs) < 0.01
    # same key+bpm + no whole-track CLAP -> only the transition term separates b from c
    for t in (a, b, c):
        t.key, t.bpm = "8A", 128.0
    m_ab = library.mixability(a, b, vectors={}, sections=secs)
    m_ac = library.mixability(a, c, vectors={}, sections=secs)
    assert m_ab > m_ac                                       # better mix-point flow ranks higher


def test_load_vectors_and_clap_similarity(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    paths = [t.path for t in library.search("", db_path=db)]
    rels = {p: Path(p).relative_to(root).as_posix() for p in paths}
    # 3-d toy vectors: p0 and p1 point the same way (cosine 1), p2 orthogonal (cosine 0)
    vp = _write_vectors(root, {
        rels[paths[0]]: [1.0, 0.0, 0.0],
        rels[paths[1]]: [2.0, 0.0, 0.0],   # different magnitude, same direction -> cosine 1
        rels[paths[2]]: [0.0, 1.0, 0.0],
    })
    library.clear_vector_cache()
    vecs = library.load_vectors(vectors_path=vp, lib_root=root, force=True)
    assert len(vecs) == 3
    import numpy as np
    assert abs(float(np.linalg.norm(next(iter(vecs.values())))) - 1.0) < 1e-5  # unit-normalized
    assert library.clap_similarity(paths[0], paths[0], vecs) == 1.0
    assert abs(library.clap_similarity(paths[0], paths[1], vecs) - 1.0) < 1e-5
    assert abs(library.clap_similarity(paths[0], paths[2], vecs)) < 1e-5
    assert library.clap_similarity(paths[0], paths[3], vecs) is None  # p3 un-embedded
    library.clear_vector_cache()


def test_load_vectors_applies_persisted_mean_centering(tmp_path):
    """When music_vectors.sqlite carries a vector_stats mean, load_vectors must subtract it and
    re-normalize (the anisotropy fix) so the PC's similarity space matches the box's centered UMAP."""
    import sqlite3, struct
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    paths = [t.path for t in library.search("", db_path=db)]
    rels = {p: Path(p).relative_to(root).as_posix() for p in paths}
    # two near-identical raw vectors (anisotropic: cosine ~1.0 before centering)
    raw = {rels[paths[0]]: [1.0, 0.10, 0.0], rels[paths[1]]: [1.0, 0.00, 0.10]}
    vp = _write_vectors(root, raw)
    mean = [1.0, 0.05, 0.05]                       # dataset-mean ~ the common component
    con = sqlite3.connect(str(vp))
    con.execute("CREATE TABLE vector_stats (k TEXT PRIMARY KEY, v BLOB)")
    con.execute("INSERT INTO vector_stats VALUES('mean', ?)",
                (struct.pack("<3f", *mean),))
    con.commit(); con.close()
    library.clear_vector_cache()
    vecs = library.load_vectors(vectors_path=vp, lib_root=root, force=True)
    import numpy as np
    # raw cosine is ~1.0 (anisotropic); after centering the two diverge sharply
    raw0 = np.array(raw[rels[paths[0]]]); raw1 = np.array(raw[rels[paths[1]]])
    raw_cos = float(raw0 @ raw1 / (np.linalg.norm(raw0) * np.linalg.norm(raw1)))
    cen_cos = library.clap_similarity(paths[0], paths[1], vecs)
    assert raw_cos > 0.97                            # collapsed before centering
    assert cen_cos < raw_cos - 0.5                   # centering restored real separation
    # the centered vectors equal (raw - mean) renormalized
    exp = raw0 - np.array(mean); exp /= np.linalg.norm(exp)
    assert np.allclose(vecs[paths[0]], exp, atol=1e-5)
    library.clear_vector_cache()


def test_virtual_artist_buckets_move_without_files_and_survive_reindex(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    md = next(t for t in library.search("", db_path=db) if "Machinedrum" in t.path)
    assert md.bucket == "dj"
    src = md.path  # the file on disk
    # reassign the artist to personal — by path so the canonical filing-folder key is used
    moved = library.set_artist_bucket(md.artist, "personal", db_path=db, path=md.path)
    assert moved >= 1
    md2 = next(t for t in library.search("", db_path=db) if "Machinedrum" in t.path)
    assert md2.bucket == "personal"          # bucket flipped
    assert md2.path == src and Path(src).exists()   # …but the FILE did not move
    assert library.get_artist_buckets(db)["machinedrum"] == "personal"
    # a re-index must NOT revert the curation back to the folder-derived bucket
    res = library.index(root=root, db_path=db)
    assert res.get("rebucketed") == 0        # already applied -> idempotent
    md3 = next(t for t in library.search("", db_path=db) if "Machinedrum" in t.path)
    assert md3.bucket == "personal"


def test_set_artist_bucket_matches_messy_tag_via_folder(tmp_path):
    """A track whose embedded tag doesn't reduce to the folder name (e.g. 'UGK (Underground
    Kingz)') must still move with the folder-named artist — the folder is the canonical key."""
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    # file filed under .../personal/UGK/ but with a messier embedded-style tag in the DB
    f = root / "music/personal/UGK" / "Big Pimpin.flac"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\x00" * 2048)
    library.index(root=root, db_path=db)
    con = library.connect(db)
    con.execute("UPDATE tracks SET artist=? WHERE path=?",
                ("UGK (Underground Kingz)", str(f)))
    con.commit(); con.close()
    moved = library.set_artist_bucket("UGK", "dj", db_path=db, path=str(f))
    assert moved >= 1
    t = next(t for t in library.search("", db_path=db) if "UGK" in t.path)
    assert t.bucket == "dj"


def test_list_buckets_reflects_present_only(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    # re-tag every artist dj/personal -> the 'mp3' folder bucket empties out
    library.seed_artist_buckets(
        {"Nia Archives": "dj", "Machinedrum": "dj", "Karol G": "personal", "Cobrah": "dj"},
        db_path=db)
    buckets = library.list_buckets(db)
    assert buckets[:2] == ["dj", "personal"]   # ordered dj, personal first
    assert "mp3" not in buckets                 # emptied -> not shown


def test_similar_tracks_ranks_and_skips_unindexed(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    tracks = library.search("", db_path=db)
    paths = [t.path for t in tracks]
    rels = {p: Path(p).relative_to(root).as_posix() for p in paths}
    vp = _write_vectors(root, {
        rels[paths[0]]: [1.0, 0.0, 0.0],
        rels[paths[1]]: [0.9, 0.1, 0.0],   # closest to seed
        rels[paths[2]]: [0.0, 1.0, 0.0],   # far
        rels[paths[3]]: [0.2, 0.9, 0.0],
        "music/Ghost/Stale.flac": [0.99, 0.01, 0.0],  # NOT in the index -> must be skipped
    })
    library.clear_vector_cache()
    seed = next(t for t in tracks if t.path == paths[0])
    sim = library.similar_tracks(seed, n=10, db_path=db, vectors_path=vp, lib_root=root)
    sim_paths = [t.path for t, _ in sim]
    assert paths[0] not in sim_paths                  # never the seed
    assert all(str(root / "music/Ghost") not in p for p in sim_paths)  # stale vector skipped
    assert sim_paths[0] == paths[1]                   # nearest neighbour ranks first
    assert sim[0][1] >= sim[-1][1]                    # similarities sorted descending
    library.clear_vector_cache()


def test_mixability_fuses_and_degrades(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    tracks = library.search("", db_path=db)
    a, b, c = tracks[0], tracks[1], tracks[2]
    # same key + same bpm for all -> only the CLAP term separates them
    for t in (a, b, c):
        t.key, t.bpm = "8A", 128.0
    vecs = {a.path: __import__("numpy").array([1.0, 0.0, 0.0], dtype="float32"),
            b.path: __import__("numpy").array([1.0, 0.0, 0.0], dtype="float32"),  # identical sound
            c.path: __import__("numpy").array([0.0, 1.0, 0.0], dtype="float32")}  # different sound
    m_ab = library.mixability(a, b, vectors=vecs)
    m_ac = library.mixability(a, c, vectors=vecs)
    assert m_ab > m_ac                                # sonically-closer pair mixes better
    assert m_ab <= 1.0 and m_ac >= 0.0
    # un-embedded track: CLAP term drops, weight redistributes over key+bpm (still a valid score)
    d = tracks[3]
    d.key, d.bpm = "8A", 128.0
    m_ad = library.mixability(a, d, vectors=vecs)     # d has no vector
    assert abs(m_ad - 1.0) < 1e-6                     # same key + same bpm, no clap -> perfect


# --- smart crates -----------------------------------------------------------
def _seed_smart_fixture(tmp_path):
    root = _make_lib(tmp_path / "lib")
    db = tmp_path / "t.db"
    library.index(root=root, db_path=db)
    paths = [t.path for t in library.search("", db_path=db)]
    con = library.connect(db)
    # p0 techno 128/8A r5 dance .8 ; p1 128/9A r3 dance .6 ; p2 90/3B r5 dance .2 ; p3 175/8A r1
    con.execute("UPDATE tracks SET bpm=128, key='8A', rating=5, danceability=0.8 WHERE path=?", (paths[0],))
    con.execute("UPDATE tracks SET bpm=128, key='9A', rating=3, danceability=0.6 WHERE path=?", (paths[1],))
    con.execute("UPDATE tracks SET bpm=90,  key='3B', rating=5, danceability=0.2 WHERE path=?", (paths[2],))
    con.execute("UPDATE tracks SET bpm=175, key='8A', rating=1, danceability=0.9 WHERE path=?", (paths[3],))
    con.commit()
    con.close()
    library.set_track_tags(paths[0], "mood", ["dark"], db_path=db)
    return db, paths


def test_smart_crate_and_conditions(tmp_path):
    db, paths = _seed_smart_fixture(tmp_path)
    spec = {"match": "all", "conditions": [
        {"field": "bpm", "op": "between", "value": [120, 135]},
        {"field": "rating", "op": ">=", "value": 4},
    ]}
    got = {t.path for t in library.evaluate_smart_crate(spec, db_path=db)}
    assert got == {paths[0]}            # p1 fails rating, p2 fails bpm, p3 fails bpm


def test_smart_crate_harmonic_and_any(tmp_path):
    db, paths = _seed_smart_fixture(tmp_path)
    # harmonic to 8A includes 8A/8B/7A/9A -> p0,p1,p3 by key; AND bpm 120-135 -> p0,p1
    spec = {"match": "all", "conditions": [
        {"field": "key", "op": "harmonic", "value": "8A"},
        {"field": "bpm", "op": "between", "value": [120, 135]},
    ]}
    assert {t.path for t in library.evaluate_smart_crate(spec, db_path=db)} == {paths[0], paths[1]}
    # match=any: rating>=5 OR danceability>=0.85 -> p0,p2 (r5) + p3 (dance .9)
    spec_any = {"match": "any", "conditions": [
        {"field": "rating", "op": ">=", "value": 5},
        {"field": "danceability", "op": ">=", "value": 0.85},
    ]}
    assert {t.path for t in library.evaluate_smart_crate(spec_any, db_path=db)} == {paths[0], paths[2], paths[3]}


def test_smart_crate_tag_and_bucket(tmp_path):
    db, paths = _seed_smart_fixture(tmp_path)
    assert {t.path for t in library.evaluate_smart_crate(
        {"conditions": [{"field": "tag", "op": "has", "value": ["mood", "dark"]}]}, db_path=db)} == {paths[0]}
    assert {t.path for t in library.evaluate_smart_crate(
        {"conditions": [{"field": "bucket", "op": "is", "value": "mp3"}]}, db_path=db)} == {paths[0]}
    # empty / invalid spec -> nothing (never the whole library)
    assert library.evaluate_smart_crate({"conditions": []}, db_path=db) == []
    assert library.evaluate_smart_crate({"conditions": [{"field": "nope", "op": "x"}]}, db_path=db) == []


def test_smart_crate_persistence(tmp_path):
    db, paths = _seed_smart_fixture(tmp_path)
    spec = {"match": "all", "conditions": [{"field": "rating", "op": ">=", "value": 4}]}
    library.save_smart_crate("Bangers", spec, db_path=db)
    assert library.list_smart_crates(db_path=db) == ["Bangers"]
    assert library.read_smart_crate("Bangers", db_path=db) == spec
    library.save_smart_crate("Bangers", {"conditions": [{"field": "bucket", "op": "is", "value": "dj"}]}, db_path=db)
    assert len(library.list_smart_crates(db_path=db)) == 1   # upsert, not duplicate
    library.delete_smart_crate("Bangers", db_path=db)
    assert library.list_smart_crates(db_path=db) == []


# --- library health (duplicates / missing / low quality) --------------------
def _make_health_lib(tmp_path):
    root = tmp_path / "lib"
    db = tmp_path / "t.db"
    for rel in ("music/dj/Artist A/Song One.flac",     # same track, lossless
                "music-mp3/Artist A/Song One.mp3",     # ...and a lossy dup (other bucket)
                "music/personal/Artist B/Unique.flac"):  # no dup
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"\x00" * 2048)
    library.index(root=root, db_path=db)
    fl = str(root / "music/dj/Artist A/Song One.flac")
    mp = str(root / "music-mp3/Artist A/Song One.mp3")
    con = library.connect(db)
    con.execute("UPDATE tracks SET duration=100, size=10000000 WHERE path=?", (fl,))  # ~800 kbps
    con.execute("UPDATE tracks SET duration=100, size=1500000 WHERE path=?", (mp,))   # ~120 kbps
    con.commit()
    con.close()
    return root, db, fl, mp


def test_find_duplicate_groups(tmp_path):
    root, db, fl, mp = _make_health_lib(tmp_path)
    groups = library.find_duplicate_groups(db_path=db)
    assert len(groups) == 1 and len(groups[0]) == 2
    assert groups[0][0].path == fl   # lossless FLAC ranked first = suggested keep
    assert groups[0][1].path == mp   # lossy copy is the redundant one


def test_find_low_quality_and_missing(tmp_path):
    root, db, fl, mp = _make_health_lib(tmp_path)
    lq = {t.path for t, _k in library.find_low_quality(db_path=db, min_kbps=256)}
    assert mp in lq and fl not in lq          # lossless never flagged; 120<256 mp3 is
    (root / "music/personal/Artist B/Unique.flac").unlink()
    missing = library.find_missing_files(db_path=db)
    assert len(missing) == 1 and missing[0].path.endswith("Unique.flac")


def test_duplicate_keep_prefers_existing_over_missing(tmp_path):
    # SMB case-folder trap: a higher-"quality" copy that's MISSING must not be the suggested keep
    root, db, fl, mp = _make_health_lib(tmp_path)
    # make the FLAC (normally the keep) missing; the mp3 still exists
    Path(fl).unlink()
    g = library.find_duplicate_groups(db_path=db)[0]
    assert g[0].path == mp     # existing mp3 is keep, the missing flac is the redundant one
    assert g[1].path == fl


def test_library_health_summary(tmp_path):
    root, db, fl, mp = _make_health_lib(tmp_path)
    h = library.library_health(db_path=db, min_kbps=256)
    assert h["redundant_copies"] == 1                 # one removable copy (the mp3)
    assert len(h["duplicate_groups"]) == 1
    assert len(h["low_quality"]) == 1 and h["missing"] == []
