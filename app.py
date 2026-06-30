"""Crate — native PySide6 desktop app for DJ set prep.

One window, two modes: a LIST (searchable table) and a MAP (CLAP-embedding UMAP), sharing one
transport, crate panel, and history. Browse your local music library, build a crate, save it as
a folder you can reopen, and export a rekordbox-ready .m3u8 + XML + local copies.
All data logic lives in library.py; this file is the Qt shell, styled in the CON//FLUENT
terminal language (see theme.py + skins/).

Run:  .venv\\Scripts\\python.exe app.py
"""
from __future__ import annotations

import random
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QThread, QTimer, Signal, QObject, QUrl, QEvent,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QComboBox, QPushButton, QTableView, QListWidget, QListWidgetItem, QLabel,
    QAbstractItemView, QFileDialog, QMessageBox, QHeaderView, QSplitter, QSizePolicy,
    QSlider, QStackedWidget, QInputDialog, QDialog, QFormLayout, QCompleter,
    QScrollArea, QFrame, QMenu, QProgressBar,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QShortcut, QKeySequence, QColor, QBrush, QFont, QIcon

import library
import theme
import map_view
import waveform_view
import tag_drawer

COLS = ["ARTIST", "TITLE", "ALBUM", "BPM", "KEY", "DANCE", "★", "COLOR", "TAGS", "CUE", "BUCKET", "FMT", "COMMENT"]
COL_W = [126, 150, 86, 44, 40, 60, 44, 44, 98, 34, 58, 40, 108]   # compact defaults; last (COMMENT) stretches to fill

# rekordbox My-Tag groups (structured) — each holds multiple values; 'free' = open keywords.
TAG_CATEGORIES = ["genre", "components", "vocal", "situation", "mood"]
# rekordbox track colour labels (the swatch names rekordbox exposes).
TRACK_COLORS = ["(none)", "pink", "red", "orange", "yellow", "green", "aqua", "blue", "purple"]
TRACK_COLOR_HEX = {
    "pink": "#ff7ab8", "red": "#ff5b5b", "orange": "#ff9f43", "yellow": "#ffd93b",
    "green": "#5bd97a", "aqua": "#48d6d6", "blue": "#5c9dff", "purple": "#b58cff",
}


def _fmt_bpm(b):
    return f"{b:.0f}" if b else ""


def _fmt_ms(ms):
    s = max(0, int(ms) // 1000)
    return f"{s // 60}:{s % 60:02d}"


def ascii_header(text: str) -> QLabel:
    """The signature CON//FLUENT '+--- LABEL --------+' frame strip (clipped to width)."""
    lbl = QLabel(f"+-- {text.upper()} " + "-" * 120 + "+")
    lbl.setObjectName("panelHeader")
    lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)  # take given width, clip the rest
    return lbl


class TrackModel(QAbstractTableModel):
    SORT_KEYS = {
        0: lambda t: (t.artist or "").lower(), 1: lambda t: (t.title or "").lower(),
        2: lambda t: (t.album or "").lower(), 3: lambda t: t.bpm or 0.0,
        4: lambda t: t.key or "", 5: lambda t: t.danceability if t.danceability is not None else -1,
        6: lambda t: t.rating or 0, 7: lambda t: (t.color or ""),
        8: lambda t: getattr(t, "tag_summary", ""), 9: lambda t: getattr(t, "cue_count", 0),
        10: lambda t: (t.bucket or "").lower(), 11: lambda t: t.ext or "",
        12: lambda t: (t.comment or "").lower(),
    }

    PLAYING_BG = QColor(180, 210, 255, 64)   # cold-blue glow for the now-playing row (distinct from grey selection)

    def __init__(self, tracks=None):
        super().__init__()
        self.tracks: list[library.Track] = tracks or []
        self.playing_path: str | None = None     # which row is currently playing (distinct from selection)

    def set_tracks(self, tracks):
        # attach the DJ-metadata summaries (tags / cue counts) for the new table columns
        summaries = library.all_tag_summaries()
        counts = library.cue_counts()
        for t in tracks:
            t.tag_summary = summaries.get(t.path, "")
            t.cue_count = counts.get(t.path, 0)
        self.beginResetModel()
        self.tracks = tracks
        self.endResetModel()

    def set_playing(self, path):
        """Mark the now-playing row (independent of selection). Refresh the old + new rows."""
        old = self.playing_path
        self.playing_path = path
        for p in (old, path):
            r = self.row_for_path(p)
            if r is not None:
                self.refresh_row(r)

    def row_for_path(self, path):
        if path is None:
            return None
        return next((i for i, t in enumerate(self.tracks) if t.path == path), None)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.tracks)

    def columnCount(self, parent=QModelIndex()):
        return len(COLS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        t = self.tracks[index.row()]
        playing = self.playing_path is not None and t.path == self.playing_path
        col = index.column()
        if role == Qt.DisplayRole:
            cues = getattr(t, "cue_count", 0)
            dance = f"{t.danceability:.2f}" if t.danceability is not None else ""
            vals = [t.artist, t.title, t.album, _fmt_bpm(t.bpm), t.key or "", dance,
                    "★" * (t.rating or 0), (t.color or ""), getattr(t, "tag_summary", ""),
                    (str(cues) if cues else ""), t.bucket, t.ext.lstrip("."), (t.comment or "")]
            if playing and col == 0:                   # ▶ marker on the playing row's artist cell
                return f"▶ {vals[0]}"
            return vals[col]
        if role == Qt.DecorationRole and col == 7 and t.color:   # COLOR column -> a swatch
            hexc = TRACK_COLOR_HEX.get(t.color)
            if hexc:
                return QColor(hexc)
        if role == Qt.TextAlignmentRole and col in (3, 5, 9):    # BPM + DANCE + CUE right-aligned
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if playing and role == Qt.BackgroundRole:
            return QBrush(self.PLAYING_BG)
        if playing and role == Qt.FontRole:
            f = QFont()
            f.setBold(True)
            return f
        return None

    def sort(self, col, order=Qt.AscendingOrder):
        key = self.SORT_KEYS.get(col)
        if not key:
            return
        self.layoutAboutToBeChanged.emit()
        self.tracks.sort(key=key, reverse=(order == Qt.DescendingOrder))
        self.layoutChanged.emit()

    def refresh_row(self, row):
        self.dataChanged.emit(self.index(row, 0), self.index(row, len(COLS) - 1))

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return COLS[section]
        return None

    def track_at(self, row) -> library.Track:
        return self.tracks[row]


class FitTable(QTableView):
    """A table whose columns ALWAYS sum to exactly the viewport width — no black band, no
    overflow off the window — while staying user-resizable. Two cases are handled:
      • window resize  -> all columns scale proportionally to the new width (_fit)
      • dragging a divider -> the dragged column keeps its new width and the *other* columns
        give/take the difference so the total stays pinned to the viewport (_redistribute)
    (COL_W just seeds the starting proportions.)"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._fitting = False
        self.horizontalHeader().sectionResized.connect(self._on_section_resized)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _fit(self):
        if self._fitting:
            return
        hh = self.horizontalHeader()
        n = hh.count()
        avail = self.viewport().width()
        cur = sum(self.columnWidth(c) for c in range(n))
        if n == 0 or cur <= 0 or avail <= 0 or abs(cur - avail) <= 1:
            return
        scale = avail / cur
        mn = hh.minimumSectionSize()
        self._fitting = True
        acc = 0
        for c in range(n - 1):
            wdt = max(mn, int(self.columnWidth(c) * scale))
            self.setColumnWidth(c, wdt)
            acc += wdt
        self.setColumnWidth(n - 1, max(mn, avail - acc))   # last col absorbs rounding -> exact fit
        self._fitting = False

    def _on_section_resized(self, idx, _old, _new):
        # user dragged a divider: pin the total back to the viewport by giving/taking the
        # difference from the OTHER columns (neighbours first), so nothing ever overflows
        # the window or leaves a black gap to the right of the last column.
        if self._fitting:
            return
        hh = self.horizontalHeader()
        n = hh.count()
        avail = self.viewport().width()
        if n <= 1 or avail <= 0:
            return
        mn = hh.minimumSectionSize()
        delta = sum(self.columnWidth(c) for c in range(n)) - avail  # >0 too wide, <0 black band
        if abs(delta) <= 1:
            return
        self._fitting = True
        # absorb from columns to the right of the dragged one first, then to the left
        order = list(range(idx + 1, n)) + list(range(idx - 1, -1, -1))
        for c in order:
            if abs(delta) <= 1:
                break
            w = self.columnWidth(c)
            new_w = max(mn, w - delta)      # delta>0 shrink, delta<0 grow
            self.setColumnWidth(c, new_w)
            delta -= (w - new_w)            # subtract what this column actually absorbed
        if delta > 1:                        # others maxed out shrinking -> cap the dragged col
            w = self.columnWidth(idx)
            self.setColumnWidth(idx, max(mn, w - delta))
        self._fitting = False


class FoldersDialog(QWidget):
    """⚙ FOLDERS — administrate the library scan roots + the crates folder, then re-index."""

    def __init__(self, host):
        super().__init__(host, Qt.Window)
        self.host = host
        self.setWindowTitle("CRATE — FOLDERS")
        self.setObjectName("root")
        self.resize(640, 420)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)
        v.addWidget(ascii_header("scan roots — folders crate indexes"))
        self.list = QListWidget()
        v.addWidget(self.list, 1)
        row = QHBoxLayout()
        add_btn = QPushButton("ADD FOLDER…")
        add_btn.clicked.connect(self._add)
        rel_btn = QPushButton("RELABEL")
        rel_btn.clicked.connect(self._relabel)
        rm_btn = QPushButton("REMOVE")
        rm_btn.clicked.connect(self._remove)
        row.addWidget(add_btn)
        row.addWidget(rel_btn)
        row.addWidget(rm_btn)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(ascii_header("crates folder — where saved crates are written"))
        crow = QHBoxLayout()
        self.crates_edit = QLineEdit()
        pick = QPushButton("…")
        pick.setFixedWidth(36)
        pick.clicked.connect(self._pick_crates)
        crow.addWidget(self.crates_edit, 1)
        crow.addWidget(pick)
        v.addLayout(crow)
        brow = QHBoxLayout()
        brow.addStretch(1)
        cancel = QPushButton("CANCEL")
        cancel.clicked.connect(self.close)
        save = QPushButton("SAVE & RE-INDEX")
        save.setObjectName("exportBtn")
        save.clicked.connect(self._save)
        brow.addWidget(cancel)
        brow.addWidget(save)
        v.addLayout(brow)
        self._load()

    def _load(self):
        cfg = library.load_config()
        self.list.clear()
        for r in cfg["scan_roots"]:
            it = QListWidgetItem(f"{r['label']:>12}   {r['path']}")
            it.setData(Qt.UserRole, dict(r))
            self.list.addItem(it)
        self.crates_edit.setText(cfg["crates_root"])

    def _add(self):
        d = QFileDialog.getExistingDirectory(self, "Add a folder to index")
        if d:
            label = Path(d).name or d
            it = QListWidgetItem(f"{label:>12}   {d}")
            it.setData(Qt.UserRole, {"label": label, "path": d})
            self.list.addItem(it)

    def _relabel(self):
        it = self.list.currentItem()
        if not it:
            return
        r = it.data(Qt.UserRole)
        new, ok = QInputDialog.getText(self, "Relabel", "Label (becomes the bucket):", text=r["label"])
        if ok and new.strip():
            r["label"] = new.strip()
            it.setData(Qt.UserRole, r)
            it.setText(f"{r['label']:>12}   {r['path']}")

    def _remove(self):
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))

    def _pick_crates(self):
        d = QFileDialog.getExistingDirectory(self, "Crates folder", self.crates_edit.text())
        if d:
            self.crates_edit.setText(d)

    def _save(self):
        roots = [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]
        if not roots:
            QMessageBox.information(self, "No folders", "Add at least one folder to index.")
            return
        # Merge into the existing config — never overwrite the whole file, or keys this dialog
        # doesn't edit (lib_root, analysis_python, skin) get dropped and sidecars resolve wrong.
        cfg = library.load_config()
        cfg["scan_roots"] = roots
        cfg["crates_root"] = self.crates_edit.text().strip()
        library.save_config(cfg)
        self.close()
        self.host._refresh_saved()
        self.host.do_index()


# ── smart crates (rule-based dynamic playlists) ────────────────────────────────────────────────
# field key -> (label, [ (op_label, op_value), ... ], value placeholder)
SMART_FIELDS = [
    ("bpm",          "BPM",          [("between", "between"), ("≥", ">="), ("≤", "<=")], "120-130"),
    ("key",          "Key",          [("harmonic", "harmonic"), ("is", "is")],            "8A"),
    ("rating",       "Rating",       [("≥", ">="), ("≤", "<="), ("is", "is")],            "4"),
    ("danceability", "Danceability", [("≥", ">="), ("≤", "<="), ("between", "between")],  "0.5"),
    ("energy",       "Energy",       [("≥", ">="), ("≤", "<="), ("between", "between")],  "0.05"),
    ("lufs",         "Loudness LUFS", [("≥", ">="), ("≤", "<="), ("between", "between")], "-9"),
    ("bucket",       "Bucket",       [("is", "is")],                                      "music"),
    ("tag",          "Tag",          [("has", "has"), ("hasn't", "not_has")],             "mood:dark"),
    ("text",         "Text",         [("contains", "contains")],                          "remix"),
    ("artist",       "Artist",       [("contains", "contains"), ("is", "is")],            "Arca"),
    ("title",        "Title",        [("contains", "contains"), ("is", "is")],            "edit"),
]
_SMART_FIELD_MAP = {k: v for k, *v in [(f[0], f[1], f[2], f[3]) for f in SMART_FIELDS]}


class _CondRow(QWidget):
    """One smart-crate condition: field + operator + value + remove. Emits a change callback so
    the dialog can live-update its match-count preview."""

    def __init__(self, on_change, on_remove, cond: dict | None = None):
        super().__init__()
        self._on_change = on_change
        self.field = QComboBox()
        for key, (label, _ops, _ph) in _SMART_FIELD_MAP.items():
            self.field.addItem(label, key)
        self.op = QComboBox()
        self.value = QLineEdit()
        self.value.setMaximumWidth(150)
        rm = QPushButton("×")
        rm.setFixedWidth(28)
        rm.setToolTip("Remove this condition")
        rm.clicked.connect(lambda: on_remove(self))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.field, 2)
        lay.addWidget(self.op, 1)
        lay.addWidget(self.value, 2)
        lay.addWidget(rm)
        self.field.currentIndexChanged.connect(self._field_changed)
        self.op.currentIndexChanged.connect(lambda *_: on_change())
        self.value.textChanged.connect(lambda *_: on_change())
        if cond:
            self._load(cond)
        else:
            self._field_changed()

    def _field_changed(self, *_):
        key = self.field.currentData()
        _label, ops, ph = _SMART_FIELD_MAP[key]
        self.op.blockSignals(True)
        self.op.clear()
        for op_label, op_val in ops:
            self.op.addItem(op_label, op_val)
        self.op.blockSignals(False)
        self.value.setPlaceholderText(ph)
        self.value.setEnabled(key not in ())   # all fields take a value here
        self._on_change()

    def _load(self, cond: dict):
        i = self.field.findData((cond.get("field") or "").lower())
        if i >= 0:
            self.field.setCurrentIndex(i)
        self._field_changed()
        j = self.op.findData((cond.get("op") or "").lower())
        if j >= 0:
            self.op.setCurrentIndex(j)
        v = cond.get("value")
        if isinstance(v, (list, tuple)):
            v = "-".join(str(x) for x in v)
        self.value.setText("" if v is None else str(v))

    def to_cond(self) -> dict | None:
        field = self.field.currentData()
        op = self.op.currentData()
        raw = self.value.text().strip()
        if not raw:
            return None
        if op == "between":
            parts = [p for p in raw.replace(",", "-").split("-") if p.strip()]
            try:
                return {"field": field, "op": op, "value": [float(parts[0]), float(parts[1])]}
            except (ValueError, IndexError):
                return None
        if field in library._SMART_NUM_FIELDS and op in (">=", "<=", "is"):
            try:
                return {"field": field, "op": op, "value": float(raw)}
            except ValueError:
                return None
        return {"field": field, "op": op, "value": raw}


class SmartCrateDialog(QDialog):
    """Create/edit a smart crate: a name, ALL/ANY match, and a list of conditions, with a live
    'matches N tracks' preview. Saving persists the spec; the caller then opens it."""

    def __init__(self, host, name: str | None = None, spec: dict | None = None):
        super().__init__(host)
        self.host = host
        self.saved_name = None
        self.deleted_name = None
        self._editing = name            # the existing crate being edited (None = brand new)
        self.setWindowTitle("Smart crate")
        self.setObjectName("root")      # picks up the skin's dark dialog background (#root)
        self.setMinimumWidth(520)
        self.name_edit = QLineEdit(name or "")
        self.name_edit.setPlaceholderText("smart crate name (e.g. peak-time bangers)")
        self.match_box = QComboBox()
        self.match_box.addItem("Match ALL conditions", "all")
        self.match_box.addItem("Match ANY condition", "any")
        if spec and spec.get("match") == "any":
            self.match_box.setCurrentIndex(1)
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(5)
        self.rows: list[_CondRow] = []
        rows_host = QWidget()
        rows_host.setLayout(self.rows_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows_host)
        scroll.setMinimumHeight(180)
        scroll.setFrameShape(QFrame.NoFrame)
        add_btn = QPushButton("+ ADD CONDITION")
        add_btn.clicked.connect(lambda: self._add_row())
        self.preview = QLabel("—")
        self.preview.setObjectName("readout")
        save = QPushButton("SAVE")
        save.setObjectName("harmonicBtn")
        save.clicked.connect(self._save)
        cancel = QPushButton("CANCEL")
        cancel.clicked.connect(self.reject)

        form = QVBoxLayout(self)
        form.setSpacing(8)
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("NAME"))
        nrow.addWidget(self.name_edit, 1)
        nrow.addWidget(self.match_box)
        form.addLayout(nrow)
        form.addWidget(scroll, 1)
        form.addWidget(add_btn)
        form.addWidget(self.preview)
        brow = QHBoxLayout()
        if self._editing:
            delete = QPushButton("DELETE")
            delete.clicked.connect(self._delete)
            brow.addWidget(delete)
        brow.addStretch(1)
        brow.addWidget(cancel)
        brow.addWidget(save)
        form.addLayout(brow)
        self.match_box.currentIndexChanged.connect(lambda *_: self._changed())

        for c in (spec or {}).get("conditions", []):
            self._add_row(c)
        if not self.rows:
            self._add_row()

    def _add_row(self, cond: dict | None = None):
        row = _CondRow(self._changed, self._remove_row, cond)
        self.rows.append(row)
        self.rows_box.addWidget(row)
        self._changed()

    def _remove_row(self, row: _CondRow):
        if row in self.rows:
            self.rows.remove(row)
            row.setParent(None)
            row.deleteLater()
        self._changed()

    def spec(self) -> dict:
        conds = [c for c in (r.to_cond() for r in self.rows) if c]
        return {"match": self.match_box.currentData(), "conditions": conds}

    def _changed(self, *_):
        try:
            n = len(library.evaluate_smart_crate(self.spec()))
            self.preview.setText(f"matches {n} track{'s' if n != 1 else ''}")
        except Exception as e:
            self.preview.setText(f"rule error: {e}")

    def _save(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.information(self, "Name needed", "Give the smart crate a name.")
            return
        if not self.spec()["conditions"]:
            QMessageBox.information(self, "No conditions", "Add at least one valid condition.")
            return
        if self._editing and self._editing != name:
            library.delete_smart_crate(self._editing)   # renamed -> drop the old entry
        library.save_smart_crate(name, self.spec())
        self.saved_name = name
        self.accept()

    def _delete(self):
        if QMessageBox.question(self, "Delete smart crate",
                                f"Delete the smart crate '{self._editing}'?") != QMessageBox.Yes:
            return
        library.delete_smart_crate(self._editing)
        self.deleted_name = self._editing
        self.reject()


class HealthDialog(QWidget):
    """♥ LIBRARY HEALTH — surfaces duplicates, missing files, and low-bitrate copies, then lets
    you trash the redundant ones in one pass (reversible — they go to TRASH, not hard-deleted)."""

    def __init__(self, host):
        super().__init__(host, Qt.Window)
        self.host = host
        self.setWindowTitle("CRATE — LIBRARY HEALTH")
        self.setObjectName("root")
        self.resize(900, 640)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)
        self.summary = QLabel("scanning…")
        self.summary.setObjectName("readout")
        v.addWidget(self.summary)

        v.addWidget(ascii_header("duplicates — same artist+title (✓ keep = best copy; rest checked to trash)"))
        self.dup_list = QListWidget()
        self.dup_list.setSelectionMode(QAbstractItemView.NoSelection)
        v.addWidget(self.dup_list, 3)

        v.addWidget(ascii_header("low bitrate — lossy copies under the threshold (re-grab lossless)"))
        self.lq_list = QListWidget()
        self.lq_list.setSelectionMode(QAbstractItemView.NoSelection)
        v.addWidget(self.lq_list, 2)

        v.addWidget(ascii_header("missing — indexed files gone from disk (checked = drop from index)"))
        self.miss_list = QListWidget()
        self.miss_list.setSelectionMode(QAbstractItemView.NoSelection)
        v.addWidget(self.miss_list, 1)

        row = QHBoxLayout()
        rescan = QPushButton("RESCAN")
        rescan.clicked.connect(self.scan)
        self.apply_btn = QPushButton("TRASH / PRUNE CHECKED")
        self.apply_btn.setObjectName("exportBtn")
        self.apply_btn.clicked.connect(self._apply)
        close = QPushButton("CLOSE")
        close.clicked.connect(self.close)
        row.addWidget(rescan)
        row.addStretch(1)
        row.addWidget(close)
        row.addWidget(self.apply_btn)
        v.addLayout(row)
        self.scan()

    @staticmethod
    def _checkable(text, path, checked):
        it = QListWidgetItem(text)
        it.setData(Qt.UserRole, path)
        it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        return it

    def scan(self):
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            h = library.library_health()
        finally:
            QApplication.restoreOverrideCursor()
        self.dup_list.clear()
        self.lq_list.clear()
        self.miss_list.clear()
        for g in h["duplicate_groups"]:
            keep = g[0]
            head = QListWidgetItem(f"▒ {keep.artist} — {keep.title}   ({len(g)} copies)")
            head.setFlags(Qt.ItemIsEnabled)
            self.dup_list.addItem(head)
            for i, t in enumerate(g):
                kb = library.estimate_kbps(t)
                tag = f"{Path(t.path).suffix.lstrip('.').upper()}{(' ' + str(round(kb)) + 'k') if kb else ''}"
                if i == 0:
                    it = QListWidgetItem(f"      ✓ KEEP  [{tag}]  {t.path}")
                    it.setFlags(Qt.ItemIsEnabled)
                else:
                    it = self._checkable(f"      trash  [{tag}]  {t.path}", t.path, True)
                self.dup_list.addItem(it)
        for t, kb in h["low_quality"]:
            self.lq_list.addItem(self._checkable(
                f"  {kb:>4}k  {Path(t.path).suffix.lstrip('.').upper():4}  {t.artist} — {t.title}", t.path, False))
        for t in h["missing"]:
            self.miss_list.addItem(self._checkable(f"  {t.path}", t.path, True))
        self.summary.setText(
            f"{len(h['duplicate_groups'])} duplicate group(s) · {h['redundant_copies']} redundant "
            f"copies · {len(h['low_quality'])} low-bitrate · {len(h['missing'])} missing")

    def _checked_paths(self):
        paths = []
        for lst in (self.dup_list, self.lq_list, self.miss_list):
            for i in range(lst.count()):
                it = lst.item(i)
                if (it.flags() & Qt.ItemIsUserCheckable) and it.checkState() == Qt.Checked:
                    paths.append(it.data(Qt.UserRole))
        return paths

    def _apply(self):
        paths = self._checked_paths()
        if not paths:
            self.summary.setText("nothing checked.")
            return
        if QMessageBox.question(
                self, "Trash / prune",
                f"Move {len(paths)} file(s) to TRASH (reversible) and drop them from the index?\n"
                f"(Missing files are just removed from the index.)") != QMessageBox.Yes:
            return
        # stop the player if it's on a file we're about to move
        if not self.host.player.source().isEmpty() and self.host.player.source().toLocalFile() in paths:
            self.host.player.stop()
            self.host.player.setSource(QUrl())
        res = library.delete_tracks(paths)
        self.host.refresh()
        self.scan()
        self.summary.setText(f"{self.summary.text()}   —   trashed/pruned {res['moved']}, "
                             f"{len(res['failed'])} failed")


class CrateWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CRATE — DJ set prep")
        self.resize(1240, 800)
        # keep the floor a touch above the column-floor crossover so the table never
        # flashes a horizontal scrollbar at the narrowest the window allows.
        self.setMinimumWidth(840)
        self.crate_paths: list[str] = []
        self.crate_tracks: dict[str, library.Track] = {}
        self.harmonic_seed = None
        self.smart_spec: dict | None = None        # active smart-crate rule lens (None = off)
        self.smart_name: str | None = None
        self.selected_track: library.Track | None = None
        self.history: list[library.Track] = []     # newest first; shared with HistoryDialog
        self._back = 0                              # how many steps back through history we've stepped
                                                    # (prev/next walk this in BOTH list + map modes)
        self.map_view = None
        self.map_view_3d = None        # software-projected 3D scatter (built lazily on 3D toggle)
        self.artist_view = None        # artist-level UMAP scatter (built lazily on ARTISTS toggle)
        self.artist_view_3d = None     # orbitable 3D artist galaxy (ARTISTS + 3D)
        self._async_thread: threading.Thread | None = None   # background file-op worker (plain thread)
        # --- playback queue (Phase C) ---
        self.playing_track: library.Track | None = None   # the track actually loaded in the player
        self.shuffle = False                              # LIST: random next instead of sequential
        self.repeat = False                               # LIST: wrap at the ends of the queue
        self.journey: set[str] = set()                    # MAP: paths already visited this walk

        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)   # topbar bleeds edge-to-edge; body is padded below
        outer.setSpacing(0)

        # --- top bar: CRATE  [LIST|MAP] .......... <readout>  ⚙ FOLDERS ---
        topbar = QWidget()
        topbar.setObjectName("topbar")
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(16, 3, 16, 3)   # inner padding so CRATE isn't jammed against the edge
        title = QLabel("CRATE")
        title.setObjectName("title")
        self.list_mode_btn = QPushButton("▤ LIST")
        self.list_mode_btn.setCheckable(True)
        self.list_mode_btn.setChecked(True)
        self.list_mode_btn.setObjectName("modeBtn")
        self.list_mode_btn.clicked.connect(lambda: self.switch_mode("list"))
        self.map_mode_btn = QPushButton("▒ MAP")
        self.map_mode_btn.setCheckable(True)
        self.map_mode_btn.setObjectName("modeBtn")
        self.map_mode_btn.clicked.connect(lambda: self.switch_mode("map"))
        self.readout = QLabel("")
        self.readout.setObjectName("readout")
        self.readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.health_btn = QPushButton("♥ HEALTH")
        self.health_btn.setToolTip("Library health — duplicates, missing files, low-bitrate copies")
        self.health_btn.clicked.connect(self.open_health)
        self.folders_btn = QPushButton("⚙ FOLDERS")
        self.folders_btn.setToolTip("Administrate which folders the library indexes")
        self.folders_btn.clicked.connect(self.open_folders)
        self.skin_box = QComboBox()
        self.skin_box.setToolTip("Skin — drop a new skins/<name>.qss in to add your own")
        for key, name in theme.list_skins():
            self.skin_box.addItem(name, key)
        cur = library.get_skin()
        i = self.skin_box.findData(cur)
        if i >= 0:
            self.skin_box.setCurrentIndex(i)
        self.skin_box.activated.connect(self._on_skin_changed)
        tb.addWidget(title)
        tb.addSpacing(12)
        tb.addWidget(self.list_mode_btn)
        tb.addWidget(self.map_mode_btn)
        tb.addStretch(1)
        tb.addWidget(self.readout)
        tb.addSpacing(10)
        tb.addWidget(self.skin_box)
        tb.addWidget(self.health_btn)
        tb.addWidget(self.folders_btn)
        outer.addWidget(topbar)

        # everything below the full-bleed topbar lives in a padded body container
        body = QWidget()
        bodyl = QVBoxLayout(body)
        bodyl.setContentsMargins(16, 6, 16, 8)
        bodyl.setSpacing(7)
        outer.addWidget(body, 1)

        # --- search / view row ---
        self.search_box = QLineEdit(placeholderText="search artist / title / album…")
        self.search_box.textChanged.connect(self.refresh)
        self.bucket_box = QComboBox()
        self._fill_bucket_box()
        self.bucket_box.currentIndexChanged.connect(self.refresh)
        view_lbl = QLabel("VIEW")
        view_lbl.setObjectName("readout")
        self.view_box = QComboBox()
        self.view_box.addItem("All songs", "all")
        self.view_box.addItem("Working crate", "crate")
        self.view_box.currentIndexChanged.connect(self.apply_view)
        self.sync_btn = QPushButton("SYNC")
        self.sync_btn.setToolTip("Pull in BPM/key analysis the pipeline has finished (no full re-index)")
        self.sync_btn.clicked.connect(self.do_sync)
        self.analyze_btn = QPushButton("ANALYZE")
        self.analyze_btn.setToolTip("Run the full analysis pipeline (BPM/key, the MuQ map + clusters, "
                                    "waveforms) for tracks that don't have it yet — on the box if "
                                    "'analysis_remote' is set, else a local analysis venv.")
        self.analyze_btn.clicked.connect(self.do_analyze)
        self.reindex_btn = QPushButton("RE-INDEX")
        self.reindex_btn.clicked.connect(self.do_index)
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self.search_box, 3)
        top.addWidget(self.bucket_box, 1)
        top.addSpacing(8)
        top.addWidget(view_lbl)
        top.addWidget(self.view_box, 1)
        top.addWidget(self.sync_btn)
        top.addWidget(self.analyze_btn)
        top.addWidget(self.reindex_btn)
        bodyl.addLayout(top)

        # --- list filter row: BPM range + harmonic key (Phase-2 analysis) ---
        self.bpm_min = QLineEdit(placeholderText="bpm min")
        self.bpm_max = QLineEdit(placeholderText="bpm max")
        for w in (self.bpm_min, self.bpm_max):
            w.setMaximumWidth(90)
            w.textChanged.connect(self.refresh)
        self.key_box = QComboBox()
        self.key_box.addItem("ALL KEYS", None)
        for n in range(1, 13):
            self.key_box.addItem(f"{n}A", f"{n}A")
        for n in range(1, 13):
            self.key_box.addItem(f"{n}B", f"{n}B")
        self.key_box.currentIndexChanged.connect(self.refresh)
        # BPM/KEY filters (LIST only) and the map controls (MAP only) share this row;
        # switch_mode() shows exactly one set.
        self.list_filters = QWidget()
        lf = QHBoxLayout(self.list_filters)
        lf.setContentsMargins(0, 0, 0, 0)
        lf.setSpacing(8)
        lbl = QLabel("BPM")
        lbl.setObjectName("readout")
        lf.addWidget(lbl)
        lf.addWidget(self.bpm_min)
        lf.addWidget(self.bpm_max)
        kl = QLabel("KEY")
        kl.setObjectName("readout")
        lf.addSpacing(8)
        lf.addWidget(kl)
        lf.addWidget(self.key_box)
        sl = QLabel("SMART")
        sl.setObjectName("readout")
        lf.addSpacing(8)
        lf.addWidget(sl)
        self.smart_box = QComboBox()
        self.smart_box.setMinimumWidth(150)
        self.smart_box.setToolTip("Smart crates — rule-based auto-crates that resolve live")
        self.smart_box.activated.connect(self._on_smart_selected)
        lf.addWidget(self.smart_box)
        self.smart_edit_btn = QPushButton("✎")
        self.smart_edit_btn.setFixedWidth(30)
        self.smart_edit_btn.setToolTip("New / edit smart crate (rule-based auto-crate)")
        self.smart_edit_btn.clicked.connect(self.open_smart_editor)
        lf.addWidget(self.smart_edit_btn)
        self._refresh_smart()
        self.map_controls = self._build_map_controls()
        self.filt_row = QWidget()
        filt = QHBoxLayout(self.filt_row)
        filt.setContentsMargins(0, 0, 0, 0)
        filt.setSpacing(8)
        filt.addWidget(self.list_filters)
        filt.addWidget(self.map_controls)
        filt.addStretch(1)
        bodyl.addWidget(self.filt_row)

        # --- center: stacked LIST/MAP (left) + crate panel (right) ---
        self.model = TrackModel()
        self.table = FitTable()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        # FitTable always sizes columns to fill the viewport exactly, so a horizontal
        # scrollbar is never wanted — suppress it (kills the rounding/scrollbar flash).
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.doubleClicked.connect(lambda *_: self.preview_selected())  # dbl-click = play
        self.table.selectionModel().selectionChanged.connect(self._on_table_select)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)
        hh = self.table.horizontalHeader()
        hh.setMinimumSectionSize(28)               # let columns shrink small when the window is narrow
        for c in range(len(COLS)):                 # user-resizable; FitTable scales them to fill width
            hh.setSectionResizeMode(c, QHeaderView.Interactive)
            self.table.setColumnWidth(c, COL_W[c])

        self.harmonic_btn = QPushButton("◇ COMPATIBLE")
        self.harmonic_btn.setObjectName("harmonicBtn")
        self.harmonic_btn.setToolTip("Tracks that mix well with the selected one — ranked by "
                                     "key + tempo + sound + transition flow. Click again to go back.")
        self.harmonic_btn.clicked.connect(self.show_harmonic)
        # CHAIN: take the selected compatible track as the next link — adds it to the crate (in order)
        # and re-ranks the list from there, so you walk a set step by step, fully in control.
        self.chain_btn = QPushButton("→ NEXT")
        self.chain_btn.setObjectName("chainBtn")
        self.chain_btn.setToolTip("Add the selected compatible track to the crate as the next link, "
                                  "then find what mixes after IT (build a set step by step).")
        self.chain_btn.clicked.connect(self.chain_next)
        self.chain_btn.setVisible(False)           # only meaningful inside the COMPATIBLE view
        # DRAFT PATH: one click sketches a whole ordered set from the selected track into the crate
        # (a starting point you then edit), shaped by the energy arc.
        self.arc_box = QComboBox()
        self.arc_box.setToolTip("Energy arc for DRAFT PATH")
        for label, data in (("→ flat", "flat"), ("↗ build up", "up"), ("↘ wind down", "down")):
            self.arc_box.addItem(label, data)
        self.draft_btn = QPushButton("DRAFT PATH")
        self.draft_btn.setObjectName("draftBtn")
        self.draft_btn.setToolTip("Sketch a full ordered set from the selected track into the crate — "
                                  "a starting point you then reorder/trim, not a finished set.")
        self.draft_btn.clicked.connect(self.draft_path)
        self.delete_btn = QPushButton("DELETE…")
        self.delete_btn.setObjectName("deleteBtn")
        self.delete_btn.setToolTip("Move selected tracks to trash (reversible)")
        self.delete_btn.clicked.connect(self.do_delete)
        # shown only in the TRASH folder view (toggled by _update_trash_actions)
        self.restore_btn = QPushButton("RESTORE")
        self.restore_btn.setToolTip("Move selected trashed tracks back into the library")
        self.restore_btn.clicked.connect(self.restore_selected)
        self.restore_btn.setVisible(False)
        self.purge_btn = QPushButton("DELETE FOREVER")
        self.purge_btn.setObjectName("deleteBtn")
        self.purge_btn.setToolTip("Permanently delete selected trashed tracks from disk (no undo)")
        self.purge_btn.clicked.connect(self.delete_forever_selected)
        self.purge_btn.setVisible(False)
        act = QHBoxLayout()
        act.setSpacing(8)
        act.addWidget(self.harmonic_btn)
        act.addWidget(self.chain_btn)
        act.addWidget(self.arc_box)
        act.addWidget(self.draft_btn)
        act.addStretch(1)
        act.addWidget(self.restore_btn)
        act.addWidget(self.purge_btn)
        act.addWidget(self.delete_btn)

        list_page = QWidget()
        lp = QVBoxLayout(list_page)
        lp.setContentsMargins(0, 0, 0, 0)
        lp.setSpacing(6)
        lp.addWidget(ascii_header("library"))
        lp.addWidget(self.table)
        lp.addLayout(act)

        # map page (built lazily on first switch to MAP)
        self.map_page = QWidget()
        self.map_layout = QVBoxLayout(self.map_page)
        self.map_layout.setContentsMargins(0, 0, 0, 0)
        self.map_layout.setSpacing(6)
        self.map_layout.addWidget(ascii_header("map · CLAP UMAP"))
        self.map_placeholder = QLabel(
            "MAP builds a UMAP from the analysis pipeline's CLAP embeddings.\n"
            "If empty, analysis hasn't run yet — run it (see SETUP.md) or hit SYNC, then try again.")
        self.map_placeholder.setObjectName("readout")
        self.map_placeholder.setAlignment(Qt.AlignCenter)
        self.map_placeholder.setWordWrap(True)
        self.map_placeholder.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # don't force window width
        self.map_layout.addWidget(self.map_placeholder, 1)

        self.stack = QStackedWidget()
        self.stack.addWidget(list_page)   # 0 = LIST
        self.stack.addWidget(self.map_page)  # 1 = MAP
        self.stack.setMinimumWidth(340)      # the list/map area can shrink this small

        # --- crate panel (right) ---
        crate_panel = self._build_crate_panel()

        split = QSplitter()
        split.addWidget(self.stack)
        split.addWidget(crate_panel)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([880, 340])
        bodyl.addWidget(split, 1)

        # --- shared bottom bar: transport + selected info + add + history ---
        self.player = QMediaPlayer()
        self.audio_out = QAudioOutput()
        self.audio_out.setVolume(0.85)
        self.player.setAudioOutput(self.audio_out)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_play_state)
        self.player.mediaStatusChanged.connect(self._on_media_status)  # auto-advance on track end
        self.transport = self._build_transport()
        bodyl.addWidget(self.transport)
        # inline TRACK inspector (tags/colour/comment) for the selected track — sits UNDER the
        # waveform/transport; replaces the old popup modal
        self.tag_drawer = tag_drawer.TagDrawer(self)
        bodyl.addWidget(self.tag_drawer)

        sc = QShortcut(QKeySequence(Qt.Key_Space), self.table, self.preview_selected)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        for keyseq in (Qt.Key_Return, Qt.Key_Enter):   # Enter on a row = add to crate
            en = QShortcut(QKeySequence(keyseq), self.table, self.add_selected)
            en.setContext(Qt.WidgetWithChildrenShortcut)
        for n in range(0, 6):                          # keys 0-5 rate the selected row(s)
            r = QShortcut(QKeySequence(str(n)), self.table, lambda n=n: self.rate_selected(n))
            r.setContext(Qt.WidgetWithChildrenShortcut)
        for n in range(1, 9):                          # keys 1-8 set hot cues (when the waveform has focus)
            hc = QShortcut(QKeySequence(str(n)), self.waveform, lambda n=n: self.drop_hot_cue(n))
            hc.setContext(Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("Ctrl+T"), self, lambda: self.tag_drawer.toggle_collapsed())  # toggle tag drawer

        self.setCentralWidget(root)
        self.status = self.statusBar()
        # in-app progress bar for long ops (index / analyze / export) — lives on the right of the
        # status bar, hidden when idle. Determinate when the op reports a step count, else marquee.
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(180)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.status.addPermanentWidget(self.progress_bar)
        self.map_controls.setVisible(False)

        # middle-mouse drag left/right = jog/scrub, parity with the map (works over the
        # table and the transport bar).
        self._jog = None
        self.table.viewport().installEventFilter(self)
        self.transport.installEventFilter(self)

        try:
            library.sync_features()  # pick up any analysis the box has finished
        except Exception:
            pass
        self._refresh_saved()
        self.refresh()
        self._maybe_prompt_index()

    # --- builders ---
    def _build_map_controls(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        lc = QLabel("CONNECT")
        lc.setObjectName("readout")
        self.mode_box = QComboBox()
        for label, val in (("None", "none"), ("Sonic", "sonic"), ("Key", "key"),
                           ("Tempo", "tempo"), ("Artist", "artist")):
            self.mode_box.addItem(label, val)
        self.mode_box.currentIndexChanged.connect(self._apply_map_mode)
        ld = QLabel("DENSITY")
        ld.setObjectName("readout")
        self.density_box = QComboBox()
        for d in map_view.DENSITY:
            self.density_box.addItem(d, map_view.DENSITY[d])
        self.density_box.setCurrentText("Sparse")
        self.density_box.currentIndexChanged.connect(self._apply_map_mode)
        lcb = QLabel("COLOR BY")
        lcb.setObjectName("readout")
        self.colorby_box = QComboBox()
        self.colorby_box.setToolTip("Colour the dots by sonic Cluster (real full-512-d HDBSCAN "
                                    "grouping), key, artist, tempo, energy, or danceability.")
        for label, val in (("Cluster", "cluster"), ("Key", "key"), ("Artist", "artist"),
                           ("Tempo", "tempo"), ("Energy", "energy"), ("Danceability", "danceability")):
            self.colorby_box.addItem(label, val)
        self.colorby_box.currentIndexChanged.connect(self._apply_color_by)
        reset_btn = QPushButton("RESET")
        reset_btn.clicked.connect(lambda: self.map_view and self.map_view.reset_layout())
        fit_btn = QPushButton("FIT")
        fit_btn.clicked.connect(self._fit_map)
        # ARTISTS grain: swap the track scatter for an artist-level UMAP (dot size = track count;
        # click an artist to filter the list). CONNECT/DENSITY only apply to the track map.
        self.artist_toggle = QPushButton("◍ ARTISTS")
        self.artist_toggle.setCheckable(True)
        self.artist_toggle.setToolTip(
            "Artist map: each dot is an artist (size = track count). Click one to filter the list.")
        self.artist_toggle.toggled.connect(self._toggle_artist_grain)
        # ◳ 3D — orbit the whole UMAP as a galaxy (CONNECT shapes become 3D: tempo helix, key wheel)
        self.threed_toggle = QPushButton("◳ 3D")
        self.threed_toggle.setCheckable(True)
        self.threed_toggle.setChecked(True)            # 3D galaxy is the default MAP view
        self.threed_toggle.setToolTip("3D galaxy: orbit (drag), zoom (wheel), click a dot to play. "
                                      "Tempo = a DNA helix, Key = a tiltable wheel.")
        self.threed_toggle.toggled.connect(self._toggle_3d)   # set checked BEFORE connecting → no eager build
        h.addWidget(lc)
        h.addWidget(self.mode_box)
        h.addWidget(ld)
        h.addWidget(self.density_box)
        h.addWidget(lcb)
        h.addWidget(self.colorby_box)
        h.addWidget(reset_btn)
        h.addWidget(fit_btn)
        h.addSpacing(8)
        h.addWidget(self.threed_toggle)
        h.addWidget(self.artist_toggle)
        return w

    def _build_crate_panel(self) -> QWidget:
        self.saved_combo = QComboBox()
        self.saved_combo.setToolTip("Saved crates — select one to open it for editing.")
        self.saved_combo.activated.connect(self._on_saved_selected)  # selecting opens it
        del_saved_btn = QPushButton("DEL")
        del_saved_btn.setToolTip("Delete the selected saved crate folder")
        del_saved_btn.clicked.connect(self.delete_saved)
        srow = QHBoxLayout()
        srow.setSpacing(6)
        srow.addWidget(self.saved_combo, 1)
        srow.addWidget(del_saved_btn)

        self.crate_list = QListWidget()
        self.crate_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        new_btn = QPushButton("NEW")
        new_btn.clicked.connect(self.new_crate)
        rm_btn = QPushButton("REMOVE")
        rm_btn.clicked.connect(self.remove_selected)
        clear_btn = QPushButton("CLEAR")
        clear_btn.clicked.connect(self.clear_crate)
        self.crate_name = QLineEdit(placeholderText="crate name (e.g. peak-set)")
        self.save_btn = QPushButton("SAVE CRATE")
        self.save_btn.setObjectName("harmonicBtn")
        self.save_btn.setToolTip("Persist this crate as a reopenable folder (copies + .m3u8)")
        self.save_btn.clicked.connect(self.do_save_crate)
        self.export_btn = QPushButton("EXPORT → REKORDBOX")
        self.export_btn.setObjectName("exportBtn")
        self.export_btn.clicked.connect(self.do_export)

        panel = QWidget()
        right = QVBoxLayout(panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(6)
        right.addWidget(ascii_header("crate"))
        right.addLayout(srow)
        right.addWidget(self.crate_list, 1)
        rrow = QHBoxLayout()
        rrow.setSpacing(6)
        rrow.addWidget(new_btn)
        rrow.addWidget(rm_btn)
        rrow.addWidget(clear_btn)
        right.addLayout(rrow)
        right.addWidget(self.crate_name)
        brow = QHBoxLayout()
        brow.setSpacing(6)
        brow.addWidget(self.save_btn)
        brow.addWidget(self.export_btn, 1)
        right.addLayout(brow)
        return panel

    def _build_transport(self) -> QWidget:
        self.prev_btn = QPushButton("◀◀")
        self.prev_btn.setFixedWidth(42)
        self.prev_btn.setToolTip("Go back to the previously played track (works in LIST and MAP)")
        self.prev_btn.clicked.connect(self.play_prev)
        self.play_btn = QPushButton("▶")
        self.play_btn.setObjectName("playBtn")
        self.play_btn.setFixedWidth(42)
        self.play_btn.clicked.connect(self.toggle_play)
        self.next_btn = QPushButton("▶▶")
        self.next_btn.setFixedWidth(42)
        self.next_btn.setToolTip("Next track (LIST: next in current sort/shuffle · MAP: nearest unplayed)")
        self.next_btn.clicked.connect(lambda: self.play_next(auto=False))
        self.shuffle_btn = QPushButton("SHUF")
        self.shuffle_btn.setCheckable(True)
        self.shuffle_btn.setObjectName("modeBtn")
        self.shuffle_btn.setToolTip("Shuffle the queue (LIST mode)")
        self.shuffle_btn.toggled.connect(self._set_shuffle)
        self.repeat_btn = QPushButton("RPT")
        self.repeat_btn.setCheckable(True)
        self.repeat_btn.setObjectName("modeBtn")
        self.repeat_btn.setToolTip("Repeat the queue at the ends (LIST mode)")
        self.repeat_btn.toggled.connect(self._set_repeat)
        self.now_label = QLabel("▶ NOW   —")
        self.now_label.setObjectName("nowLabel")
        self.now_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # clip long text, don't force window width
        self.now_label.setMinimumWidth(0)
        self.now_label.setToolTip("The track currently playing (never changes when you browse)")
        self.waveform = waveform_view.WaveformWidget(self)   # colored waveform replaces the seek bar
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("readout")
        self.add_btn = QPushButton("＋ ADD")
        self.add_btn.setObjectName("harmonicBtn")
        self.add_btn.setToolTip("Add the selected / now-playing track to the crate")
        self.add_btn.clicked.connect(self.add_current)
        vol_lbl = QLabel("VOL")
        vol_lbl.setObjectName("readout")
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(int(self.audio_out.volume() * 100))
        self.vol_slider.setFixedWidth(84)
        self.vol_slider.valueChanged.connect(lambda v: self.audio_out.setVolume(v / 100))
        self.history_btn = QPushButton("▤")
        self.history_btn.setToolTip("History — recently played (double-click an entry to replay)")
        self.history_btn.setFixedWidth(34)
        self.history_btn.clicked.connect(self.open_history)
        tr = QHBoxLayout()
        tr.setContentsMargins(2, 6, 2, 0)
        tr.setSpacing(6)
        tr.addWidget(self.prev_btn)
        tr.addWidget(self.play_btn)
        tr.addWidget(self.next_btn)
        tr.addWidget(self.shuffle_btn)
        tr.addWidget(self.repeat_btn)
        tr.addWidget(self.now_label, 6)
        tr.addWidget(self.time_label)
        tr.addWidget(self.add_btn)
        tr.addWidget(vol_lbl)
        tr.addWidget(self.vol_slider)
        tr.addWidget(self.history_btn)
        # second line: INSPECT — whatever you're selecting (LIST) or hovering (MAP); never the player
        self.inspect_label = QLabel("◇ INSPECT   —")
        self.inspect_label.setObjectName("inspectLabel")
        self.inspect_label.setMinimumHeight(20)
        self.inspect_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)  # take width, never clip vertically
        self.inspect_label.setToolTip("The track you're looking at — select a row or hover a dot")
        outer = QVBoxLayout()
        outer.setContentsMargins(2, 6, 2, 6)
        outer.setSpacing(4)
        outer.addWidget(self.waveform)      # full-width colored waveform = the seek surface
        outer.addLayout(tr)
        outer.addWidget(self.inspect_label)
        trw = QWidget()
        trw.setObjectName("transport")
        trw.setLayout(outer)
        return trw

    # --- modes ---
    def switch_mode(self, mode):
        is_map = mode == "map"
        self.list_mode_btn.setChecked(not is_map)
        self.map_mode_btn.setChecked(is_map)
        self.map_controls.setVisible(is_map)
        self.list_filters.setVisible(not is_map)
        if is_map:
            self._ensure_map()
            self.stack.setCurrentIndex(1)
            self._show_active_map()
            # a fresh forward-only walk, seeded by whatever is currently playing
            self.journey = {self.playing_track.path} if self.playing_track else set()
        else:
            self.stack.setCurrentIndex(0)
        self._back = 0                          # mode switch resets the history walk cursor
        self.shuffle_btn.setEnabled(not is_map)
        self.repeat_btn.setEnabled(not is_map)
        self.apply_view()

    def _ensure_map(self) -> bool:
        if self.map_view is not None:
            return True
        try:
            data = library.tracks_with_coords()
        except Exception as e:
            self.status.showMessage(f"MAP ERROR: {e}")
            data = []
        if not data:
            return False
        self.map_view = map_view.MapView(data, self)
        self.map_view.set_color_mode(self.colorby_box.currentData())   # match the COLOR BY selector
        self.map_layout.removeWidget(self.map_placeholder)
        self.map_placeholder.hide()
        self.map_layout.addWidget(self.map_view, 1)
        if self.playing_track:                 # reflect current playback + trail on the fresh map
            self.map_view.set_playing(self.playing_track.path)
            self.map_view.set_trail([x.path for x in reversed(self.history)][-40:])
        return True

    def _apply_map_mode(self, *_):
        if self.map_view:
            self.map_view.set_mode(self.mode_box.currentData(), self.density_box.currentData())
        if self.map_view_3d:
            self.map_view_3d.set_mode(self.mode_box.currentData(), self.density_box.currentData())

    def _apply_color_by(self, *_):
        if self.map_view:
            self.map_view.set_color_mode(self.colorby_box.currentData())
        if self.map_view_3d:
            self.map_view_3d.set_color_mode(self.colorby_box.currentData())

    def _active_map(self):
        """The map widget the (3D?, artist?) toggle combo currently shows (no building)."""
        three, art = self.threed_toggle.isChecked(), self.artist_toggle.isChecked()
        if three and art:
            return self.artist_view_3d or self.artist_view or self.map_view
        if three:
            return self.map_view_3d or self.map_view
        if art:
            return self.artist_view or self.map_view
        return self.map_view

    def _sync_map_controls(self):
        """CONNECT + DENSITY apply to the track maps (2D + 3D) but not the artist grain."""
        art = self.artist_toggle.isChecked()
        self.mode_box.setEnabled(not art)
        self.density_box.setEnabled(not art)

    def _show_active_map(self):
        """Show the right map for the (3D?, artist?) toggle combo — 2D track / 3D track / 2D artist /
        3D artist — building it lazily, hiding the others, and falling back gracefully if a variant's
        data isn't built yet (3D artist → 2D artist → 2D track)."""
        three, art = self.threed_toggle.isChecked(), self.artist_toggle.isChecked()
        target = None
        if three and art:
            if self._ensure_artist_map3d():
                target = self.artist_view_3d
            elif self._ensure_artist_map():
                target = self.artist_view
        elif three:
            if self._ensure_map3d():
                target = self.map_view_3d
        elif art:
            if self._ensure_artist_map():
                target = self.artist_view
        if target is None and self._ensure_map():
            target = self.map_view
        for v in (self.map_view, self.map_view_3d, self.artist_view, self.artist_view_3d):
            if v is not None and v is not target:
                v.hide()
        if target is not None:
            target.show()
        self._sync_map_controls()
        return target

    def _fit_map(self):
        v = self._active_map()
        if v:
            v.fit()

    def _ensure_map3d(self) -> bool:
        if self.map_view_3d is not None:
            return True
        try:
            data = library.tracks_with_coords3d()
        except Exception as e:
            self.status.showMessage(f"3D MAP ERROR: {e}")
            data = []
        if not data:
            self.status.showMessage("No 3D map yet — rebuild analysis (umap_music.py writes coords3d).")
            return False
        self.map_view_3d = map_view.MapView3D(data, self)
        self.map_view_3d.set_color_mode(self.colorby_box.currentData())
        self.map_view_3d.set_mode(self.mode_box.currentData(), self.density_box.currentData())
        if self.map_placeholder.isVisible():
            self.map_layout.removeWidget(self.map_placeholder)
            self.map_placeholder.hide()
        self.map_layout.addWidget(self.map_view_3d, 1)
        if self.playing_track:
            self.map_view_3d.set_playing(self.playing_track.path)
            self.map_view_3d.set_trail([x.path for x in reversed(self.history)][-40:])
        return True

    def _toggle_3d(self, checked):
        """Toggle the orbitable 3D galaxy for whichever grain is active (track or artist)."""
        if self.stack.currentIndex() != 1:             # only re-show when we're actually in MAP mode
            return
        self._show_active_map()
        self._apply()                                  # reflect the active filter subset
        self.status.showMessage("3D galaxy — drag to orbit, wheel to zoom." if checked else "2D map.")

    def _ensure_artist_map(self) -> bool:
        if self.artist_view is not None:
            return True
        try:
            data = library.artists_with_coords()
        except Exception as e:
            self.status.showMessage(f"ARTIST MAP ERROR: {e}")
            data = []
        if not data:
            self.status.showMessage("No artist map yet — run analysis/umap_artists.py (see SETUP.md).")
            return False
        self.artist_view = map_view.ArtistMapView(data, self)
        if self.map_placeholder.isVisible():
            self.map_layout.removeWidget(self.map_placeholder)
            self.map_placeholder.hide()
        self.map_layout.addWidget(self.artist_view, 1)
        return True

    def _ensure_artist_map3d(self) -> bool:
        if self.artist_view_3d is not None:
            return True
        try:
            data = library.artists_with_coords3d()
        except Exception as e:
            self.status.showMessage(f"3D ARTIST MAP ERROR: {e}")
            data = []
        if not data:
            self.status.showMessage(
                "No 3D artist map yet — rebuild analysis (umap_artists.py writes artists3d).")
            return False
        self.artist_view_3d = map_view.ArtistMapView3D(data, self)
        if self.map_placeholder.isVisible():
            self.map_layout.removeWidget(self.map_placeholder)
            self.map_placeholder.hide()
        self.map_layout.addWidget(self.artist_view_3d, 1)
        return True

    def _toggle_artist_grain(self, checked):
        """Toggle the artist-level grain (2D, or the 3D galaxy when ◳ 3D is also on)."""
        if self.stack.currentIndex() != 1:
            return
        self._show_active_map()
        self._apply()
        if checked:
            v = self._active_map()
            n = len(getattr(v, "artists", []))
            self.status.showMessage(f"ARTIST MAP — {n} artists. Click one to filter the list.")

    def inspect_artist(self, name, n):
        """Hover readout for the artist map (INSPECT channel uses Tracks, so route this to status)."""
        self.status.showMessage(f"◍ {name}  ·  {n} track{'s' if n != 1 else ''}  — click to filter")

    def filter_to_artist(self, name):
        """Click an artist on the artist map → show that artist's tracks in the LIST."""
        self.view_box.setCurrentIndex(0)        # all songs (not the working-crate lens)
        self.search_box.setText(name)           # textChanged → refresh() filters the table
        self.switch_mode("list")
        self.status.showMessage(f'Filtered to "{name}" — clear the search box to go back.')

    # --- data / view ---
    def _maybe_prompt_index(self):
        if not self.model.tracks:
            self.status.showMessage("LIBRARY NOT INDEXED — click RE-INDEX to scan your folders.")

    def _update_readout(self, n):
        self.readout.setText(f"{n} TRACKS")

    def _bpm_range(self):
        try:
            lo = float(self.bpm_min.text()) if self.bpm_min.text().strip() else None
            hi = float(self.bpm_max.text()) if self.bpm_max.text().strip() else None
        except ValueError:
            return None
        if lo is None and hi is None:
            return None
        return (lo if lo is not None else 0.0, hi if hi is not None else 999.0)

    def working_tracks(self) -> list[library.Track]:
        return [self.crate_tracks[p] for p in self.crate_paths if p in self.crate_tracks]

    def refresh(self):
        self.harmonic_seed = None          # any search/filter change exits the harmonic view
        self.smart_spec = None             # ...and the smart-crate lens
        self.smart_name = None
        if getattr(self, "smart_box", None) and self.smart_box.currentIndex() != 0:
            self.smart_box.blockSignals(True)
            self.smart_box.setCurrentIndex(0)
            self.smart_box.blockSignals(False)
        self._apply()

    def _apply(self):
        bucket = self.bucket_box.currentData()
        in_trash = bucket == "trash"
        try:
            if in_trash:                              # TRASH folder: browse/play trashed files
                tracks = library.quarantine_tracks()
            elif self.view_box.currentData() == "crate":
                tracks = self.working_tracks()
            elif self.harmonic_seed:
                tracks = library.harmonic_matches(self.harmonic_seed)
            elif self.smart_spec:
                tracks = library.evaluate_smart_crate(self.smart_spec)
            else:
                tracks = library.search(self.search_box.text().strip(),
                                        bucket=bucket,
                                        bpm_range=self._bpm_range(),
                                        key=self.key_box.currentData(), limit=2000)
        except Exception as e:
            self.status.showMessage(f"SEARCH ERROR: {e}")
            return
        self.model.set_tracks(tracks)
        # the MAP mirrors the LIST filter: bucket (dj/personal), search, BPM/key, crate, smart and
        # harmonic lenses all subset the dots. No filter at all = show the whole map (None). Apply to
        # BOTH the 2D and 3D track maps independently — 3D is the default, so map_view (2D) may never
        # have been built; gating the whole block on `map_view is not None` left 3D unfiltered.
        filtered = (in_trash or bucket is not None
                    or self.view_box.currentData() == "crate"
                    or bool(self.harmonic_seed) or bool(self.smart_spec)
                    or bool(self.search_box.text().strip())
                    or self.key_box.currentData() is not None
                    or self._bpm_range() != (None, None))
        subset = {t.path for t in tracks} if filtered else None
        if self.map_view is not None:
            self.map_view.set_subset(subset)
        if self.map_view_3d is not None:
            self.map_view_3d.set_subset(subset)
        self._update_readout(len(tracks))
        self._update_trash_actions(in_trash)
        if getattr(self, "harmonic_btn", None):
            self.harmonic_btn.setText("← BACK TO LIBRARY" if self.harmonic_seed else "◇ COMPATIBLE")
        if getattr(self, "chain_btn", None):
            self.chain_btn.setVisible(bool(self.harmonic_seed))   # "→ NEXT" only inside COMPATIBLE
        if in_trash:
            self.status.showMessage(
                f"TRASH — {len(tracks)} track(s). Play to preview, then RESTORE or DELETE FOREVER "
                f"(select all to empty).")
        elif self.view_box.currentData() == "crate":
            self.status.showMessage(f"VIEW: working crate — {len(tracks)} tracks")
        elif self.harmonic_seed:
            s = self.harmonic_seed
            self.status.showMessage(
                f"◇ COMPATIBLE with {s.artist} — {s.title}  [{s.key} · {_fmt_bpm(s.bpm)} BPM] "
                f"— {len(tracks)} matches, ranked by key+tempo+sound+transition. "
                f"→ NEXT to chain, or DRAFT PATH for a full set (search to exit)")
        elif self.smart_spec:
            self.status.showMessage(
                f"⚡ SMART: {self.smart_name} — {len(tracks)} tracks (live rule; ✎ to edit, "
                f"select-all + ＋ADD to push into the working crate)")
        else:
            self.status.showMessage(f"{len(tracks)} tracks shown")

    def apply_view(self, *_):
        """VIEW lens drives BOTH modes: All songs, or just the working crate's tracks. The map
        subset is set inside _apply() (which mirrors every active filter), so nothing extra here."""
        self._apply()

    def show_harmonic(self):
        if self.harmonic_seed:                 # already in the harmonic view -> go back
            self.harmonic_seed = None
            self._apply()
            return
        t = self.selected_track or self._selected_track()
        if not t:
            self.status.showMessage("Select a track first to find harmonic matches.")
            return
        if not t.key or not t.bpm:
            self.status.showMessage("Selected track has no BPM/key yet — run analysis, then SYNC.")
            return
        self.harmonic_seed = t
        self.view_box.blockSignals(True)
        self.view_box.setCurrentIndex(0)   # harmonic is an all-songs lens
        self.view_box.blockSignals(False)
        self._apply()

    def chain_next(self):
        """Walk the set one link at a time: add the selected compatible track to the working crate
        (in order), then re-seed COMPATIBLE from it so the list re-ranks for the next pick. The DJ
        chooses every step — the crate IS the growing sequence."""
        if not self.harmonic_seed:
            self.status.showMessage("Open ◇ COMPATIBLE first, then pick a track and hit → NEXT.")
            return
        t = self.selected_track or self._selected_track()
        if not t or t.path == self.harmonic_seed.path:
            self.status.showMessage("Select one of the compatible tracks to chain to.")
            return
        if self.harmonic_seed.path not in self.crate_paths:    # seed the crate with the first track
            self.add_track(self.harmonic_seed)
        self.add_track(t)
        self.harmonic_seed = t                                 # re-rank from the new last link
        self._apply()
        self.status.showMessage(f"Chained → {t.artist} — {t.title}  ({len(self.crate_paths)} in crate). "
                                f"Pick the next link, or DRAFT PATH to auto-extend.")

    def draft_path(self):
        """Sketch a whole ordered set from the selected (or COMPATIBLE-seed) track into the working
        crate — a starting point the DJ edits, shaped by the chosen energy arc. Reuses the crate as
        the sequence surface, so SAVE / EXPORT / rekordbox XML all work on the draft."""
        seed = self.harmonic_seed or self.selected_track or self._selected_track()
        if not seed:
            self.status.showMessage("Select a track to draft a path from.")
            return
        if not seed.key or not seed.bpm:
            self.status.showMessage("Seed track has no BPM/key yet — run analysis, then SYNC.")
            return
        arc = self.arc_box.currentData()
        try:
            path = library.build_path(seed, length=12, energy=arc)
        except Exception as e:
            self.status.showMessage(f"DRAFT PATH error: {e}")
            return
        if len(path) < 2:
            self.status.showMessage("Not enough compatible tracks to draft a path from this seed.")
            return
        added = sum(self.add_track(t) for t in path)           # append in order, skipping dups
        self.harmonic_seed = None                              # leave the COMPATIBLE lens
        self.view_box.blockSignals(True)
        self.view_box.setCurrentIndex(self.view_box.findData("crate"))   # show the drafted sequence
        self.view_box.blockSignals(False)
        self._apply()
        arc_label = {"flat": "flat", "up": "building", "down": "winding down"}.get(arc, arc)
        self.status.showMessage(f"Drafted a {len(path)}-track path ({arc_label}); added {added} → crate. "
                                f"Reorder/trim it, then SAVE or EXPORT.")

    # --- smart crates -------------------------------------------------------
    def _refresh_smart(self, select: str | None = None):
        """Repopulate the smart-crate dropdown (header + saved names); optionally select one."""
        self.smart_box.blockSignals(True)
        self.smart_box.clear()
        self.smart_box.addItem("— smart —", None)
        for name in library.list_smart_crates():
            self.smart_box.addItem(f"⚡ {name}", name)
        if select:
            i = self.smart_box.findData(select)
            if i >= 0:
                self.smart_box.setCurrentIndex(i)
        self.smart_box.blockSignals(False)

    def _on_smart_selected(self, idx):
        name = self.smart_box.itemData(idx)
        if not name:                                   # the "— smart —" header clears the lens
            self.smart_spec = None
            self.smart_name = None
            self._apply()
            return
        spec = library.read_smart_crate(name)
        if spec is None:
            self.status.showMessage(f"Smart crate '{name}' could not be read.")
            return
        self.harmonic_seed = None                      # smart is an all-songs lens, like harmonic
        self.smart_spec = spec
        self.smart_name = name
        self.bucket_box.blockSignals(True)
        self.bucket_box.setCurrentIndex(0)             # smart rule owns the bucket filter
        self.bucket_box.blockSignals(False)
        self.view_box.blockSignals(True)
        self.view_box.setCurrentIndex(0)
        self.view_box.blockSignals(False)
        self._apply()

    def open_smart_editor(self):
        """New smart crate, or edit the active one. On save, select + apply it."""
        name = self.smart_name
        spec = self.smart_spec
        dlg = SmartCrateDialog(self, name=name, spec=spec)
        result = dlg.exec()
        if result == QDialog.Accepted and dlg.saved_name:
            self._refresh_smart(select=dlg.saved_name)
            self._on_smart_selected(self.smart_box.currentIndex())
        elif dlg.deleted_name:                         # deleted from inside the editor
            self.smart_spec = None
            self.smart_name = None
            self._refresh_smart()
            self._apply()

    def _run_async(self, fn, on_done, busy_msg, steps=None):
        """Run a blocking callable off the UI thread, with the in-app progress bar.

        `steps` = how many progress ticks to expect (e.g. the analysis pipeline has 6); when given
        the bar is determinate (step n of steps), otherwise it's a marquee (we don't know how long).

        WHY a plain thread + a poll timer (not QThread + signals): on this PySide6 / Python 3.14
        build, cross-thread signal delivery to a Python slot is broken — Qt invokes the slot on the
        EMITTING (worker) thread even with an explicit Qt.QueuedConnection (verified). A worker-run
        slot that touches the status bar / progress bar is GUI access off the main thread → a native
        access-violation crash (this was the every-time SAVE-crate crash, caught by faulthandler in
        the worker mid-progress-lambda). So the job runs on a bare threading.Thread that touches ZERO
        Qt — it only writes plain attributes — and EVERY GUI update happens here on the main thread,
        driven by a QTimer poll.
        """
        if self._async_thread is not None and self._async_thread.is_alive():
            self.status.showMessage("Busy — wait for the current operation to finish.")
            return
        self._busy(True)
        self.status.showMessage(busy_msg)
        self._busy_msg = busy_msg
        self.progress_bar.setRange(0, steps if steps else 0)   # (0,0) = indeterminate marquee
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._on_done_cb = on_done
        self._result, self._err = None, None
        self._progress = (0, "")           # (n, label) — written by the worker, read by the poll timer
        self._async_done = False

        def work():                        # runs on a plain thread; MUST NOT touch any Qt object
            try:
                self._result = fn(lambda n, l: setattr(self, "_progress", (n, l)))
            except Exception as e:         # surface to the UI rather than crash
                self._err = f"{type(e).__name__}: {e}"
            finally:
                self._async_done = True

        self._async_thread = threading.Thread(target=work, daemon=True)
        self._poll_timer = QTimer(self)    # owned by this window → its timeout runs on the GUI thread
        self._poll_timer.timeout.connect(self._poll_async)
        self._poll_timer.start(80)
        self._async_thread.start()

    def _busy(self, busy):
        for b in (self.reindex_btn, self.export_btn, self.save_btn, self.analyze_btn):
            b.setEnabled(not busy)

    def _poll_async(self):
        """GUI-thread poll of the background job (see _run_async): paint progress, then on completion
        tear down + fire the error dialog / on_done callback — all safely on the main thread."""
        n, label = self._progress
        if label:
            self.status.showMessage(f"{self._busy_msg}  {label}")
            self.progress_bar.setValue(n)
        if not self._async_done:
            return
        self._poll_timer.stop()
        self._async_thread = None
        self._busy(False)
        self.progress_bar.setVisible(False)
        if self._err:
            QMessageBox.warning(self, "Error", self._err)
            self.status.showMessage(self._err)
        elif self._on_done_cb:
            self._on_done_cb(self._result)

    def do_sync(self):
        try:
            n = library.sync_features()
        except Exception as e:
            self.status.showMessage(f"SYNC ERROR: {e}")
            return
        self.refresh()
        self.status.showMessage(f"SYNCED analysis into {n} tracks")

    def do_index(self):
        self._run_async(
            lambda progress: library.index_from_config(progress=lambda n, l: progress(n, l)),
            self._on_indexed, "INDEXING…")

    def do_analyze(self):
        """Run the full analysis pipeline (BPM/key + MuQ map + clusters + waveforms) off-thread, then
        SYNC the results in. Two modes:
        - personal rig: if 'analysis_remote' is set in crate_config.json, refresh the box over SSH
          (the box does all compute; the PC reads the sidecars it writes via Z:);
        - shareable build: otherwise run the local heavy analysis venv.
        If neither is available, explain how to set one up rather than failing silently."""
        remote = library.analysis_remote_config()
        if remote:
            self._run_async(
                lambda progress: library.run_analysis_remote(
                    progress=lambda n, l: progress(n, l), cfg=remote),
                self._on_analyzed, f"REFRESHING ANALYSIS ON {remote['ssh']}…", steps=6)
            return
        if not library.analysis_available():
            QMessageBox.information(
                self, "Analysis environment needed",
                "In-app analysis needs the heavy analysis environment (torch + librosa + "
                "transformers + umap, ~2 GB) — it's deliberately kept out of the app's light venv.\n\n"
                "Set it up once:\n"
                "  python -m venv analysis/.venv\n"
                "  analysis/.venv/Scripts/pip install -r analysis/requirements-analysis.txt\n"
                "  (plus the torch build for your GPU — see that file)\n\n"
                "or point 'analysis_python' in crate_config.json at an interpreter that has them.\n\n"
                "To run analysis on a separate box instead, set 'analysis_remote' (ssh host) in "
                "crate_config.json; or if it runs there on its own, just hit SYNC after it finishes.")
            return
        root = library.LIB_ROOT
        self._run_async(
            lambda progress: library.run_analysis(root=root, progress=lambda n, l: progress(n, l)),
            self._on_analyzed, "ANALYZING (this can take a while)…", steps=6)

    def _on_analyzed(self, res):
        library.clear_vector_cache()       # new vectors/sections on disk -> drop the cached snapshot
        n = 0
        try:
            n = library.sync_features()    # pull the fresh BPM/key/energy into the index
        except Exception:
            pass
        self.map_view = None               # rebuild the map from the new UMAP on next MAP switch
        self.refresh()
        if not res or not res.get("ok"):
            tail = "\n".join((res or {}).get("log", [])[-6:])
            QMessageBox.warning(self, "Analysis stopped",
                                f"The analysis pipeline exited with an error.\n\n{tail}")
            self.status.showMessage("ANALYZE failed — see the dialog.")
        else:
            self.status.showMessage(f"ANALYZE complete — synced {n} tracks. "
                                    f"Switch to MAP to see the updated embedding map.")

    def _fill_bucket_box(self):
        """(Re)populate the bucket filter from the buckets actually present (dj/personal/…), keeping
        the current selection. ALL + virtual buckets + TRASH."""
        cur = self.bucket_box.currentData()
        self.bucket_box.blockSignals(True)
        self.bucket_box.clear()
        self.bucket_box.addItem("ALL", None)
        for label in library.list_buckets():
            self.bucket_box.addItem(label.upper(), label)
        self.bucket_box.addItem("🗑 TRASH", "trash")   # trashed tracks, browsable/playable; not in ALL
        idx = self.bucket_box.findData(cur)
        self.bucket_box.setCurrentIndex(idx if idx >= 0 else 0)
        self.bucket_box.blockSignals(False)

    def _on_indexed(self, res):
        self._fill_bucket_box()   # buckets may have changed (virtual re-tagging / new artists)
        self.refresh()
        if res.get("missing_roots"):
            QMessageBox.warning(
                self, "Folders unreachable",
                "These scan folders couldn't be reached, so nothing was pruned (your index and "
                "ratings are safe):\n\n" + "\n".join(res["missing_roots"]) +
                "\n\nReconnect the drive/folder and RE-INDEX again.")
        self.status.showMessage(
            f"INDEXED {res['total']} tracks "
            f"(+{res['added']} new · {res['updated']} changed · {res['removed']} removed"
            f"{'' if res.get('pruned', True) else ' · prune skipped (unreachable folder)'})")

    def open_folders(self):
        self._folders = FoldersDialog(self)
        self._folders.show()

    def open_health(self):
        self._health = HealthDialog(self)
        self._health.show()

    def _on_skin_changed(self, _idx):
        key = self.skin_box.currentData()
        applied = theme.apply(QApplication.instance(), key)   # restyles the whole app live
        library.set_skin(applied)
        self.status.showMessage(f"Skin: {self.skin_box.currentText()}")

    # --- crate ops ---
    def add_track(self, t) -> bool:
        if t.path in self.crate_paths:
            return False
        self.crate_paths.append(t.path)
        self.crate_tracks[t.path] = t
        self.crate_list.addItem(QListWidgetItem(f"{t.artist} — {t.title}  [{t.bucket}]"))
        self._crate_changed()
        return True

    def remove_track(self, path) -> bool:
        if path in self.crate_paths:
            i = self.crate_paths.index(path)
            del self.crate_paths[i]
            self.crate_tracks.pop(path, None)
            self.crate_list.takeItem(i)
            self._crate_changed()
            return True
        return False

    def _crate_changed(self):
        """Keep the map + the working-crate lens in sync after any crate mutation."""
        if self.map_view:
            self.map_view._restyle_all()
        if self.view_box.currentData() == "crate":
            self.apply_view()

    def add_selected(self):
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        added = sum(self.add_track(self.model.track_at(r)) for r in rows)
        self.status.showMessage(f"Added {added} → crate ({len(self.crate_paths)} total)")

    def add_current(self):
        if self.stack.currentIndex() == 1:              # MAP/3D: add what's PLAYING (the dot you clicked),
            t = self.playing_track or self.selected_track   # not whatever you last hovered over
        else:
            t = self.selected_track or self.playing_track   # LIST: prefer the selected row
        if not t:
            self.status.showMessage("Select or play a track first, then ＋ADD.")
            return
        if self.add_track(t):
            self.status.showMessage(f"Added → crate ({len(self.crate_paths)} total)")
        else:
            self.status.showMessage("Already in the crate.")

    def remove_selected(self):
        for row in sorted((self.crate_list.row(i) for i in self.crate_list.selectedItems()),
                          reverse=True):
            path = self.crate_paths[row]
            self.crate_list.takeItem(row)
            del self.crate_paths[row]
            self.crate_tracks.pop(path, None)
        self._crate_changed()

    def clear_crate(self):
        self.crate_list.clear()
        self.crate_paths.clear()
        self.crate_tracks.clear()
        self._crate_changed()

    def new_crate(self):
        self.clear_crate()
        self.crate_name.clear()
        self.status.showMessage("New empty crate.")

    # --- saved crates (folders) ---
    def _refresh_saved(self):
        self.saved_combo.blockSignals(True)
        self.saved_combo.clear()
        self.saved_combo.addItem("— saved crates —", None)
        for name, n, _mt in library.list_crates():
            self.saved_combo.addItem(f"{name}  ({n})", name)
        self.saved_combo.blockSignals(False)

    def _on_saved_selected(self, idx):
        if self.saved_combo.itemData(idx) is not None:   # ignore the "— saved crates —" header
            self.open_saved()

    def open_saved(self):
        name = self.saved_combo.currentData()
        if not name:
            self.status.showMessage("Pick a saved crate to open.")
            return
        tracks = library.read_crate(name)
        self.clear_crate()
        for t in tracks:
            self.add_track(t)
        self.crate_name.setText(name)
        self.view_box.setCurrentIndex(1)   # focus the opened crate (VIEW = working crate)
        self.apply_view()
        self.status.showMessage(f"Opened crate '{name}' ({len(tracks)} tracks)")

    def delete_saved(self):
        name = self.saved_combo.currentData()
        if not name:
            return
        if QMessageBox.question(self, "Delete crate", f"Delete the saved crate '{name}'?\n"
                                "(removes its folder; your library is untouched)",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        library.delete_crate(name)
        self._refresh_saved()
        self.status.showMessage(f"Deleted saved crate '{name}'")

    def do_save_crate(self):
        if not self.crate_paths:
            QMessageBox.information(self, "Empty crate", "Add some tracks first.")
            return
        name = self.crate_name.text().strip()
        if not name:
            name, ok = QInputDialog.getText(self, "Save crate", "Crate name:")
            if not ok or not name.strip():
                return
            name = name.strip()
            self.crate_name.setText(name)
        paths = list(self.crate_paths)
        self._run_async(
            lambda progress: library.save_crate(name, paths, progress=lambda n, l: progress(n, l)),
            self._on_saved, "SAVING CRATE…")

    def _on_saved(self, res):
        self._refresh_saved()
        # reselect the just-saved crate in the dropdown
        idx = self.saved_combo.findData(library._sanitize(self.crate_name.text().strip()))
        if idx < 0:
            idx = self.saved_combo.findText(self.crate_name.text().strip(), Qt.MatchStartsWith)
        if idx >= 0:
            self.saved_combo.setCurrentIndex(idx)
        self.status.showMessage(
            f"SAVED {res['copied']} tracks → {res['dest']}  (+ rekordbox .xml + .m3u8)")

    def do_export(self):
        if not self.crate_paths:
            QMessageBox.information(self, "Empty crate", "Add some tracks first.")
            return
        name = self.crate_name.text().strip() or "crate"
        dest_root = QFileDialog.getExistingDirectory(
            self, "Export destination", str(library.DEFAULT_EXPORT_ROOT))
        if not dest_root:
            return
        paths = list(self.crate_paths)

        def _do(progress):
            # copy files + write the simple .m3u8, then the rekordbox .xml carrying the full prep
            # (BPM/key/rating/colour/tags/cues). The xml re-uses the already-copied files.
            m = library.export(paths, name, export_root=Path(dest_root),
                               progress=lambda n, l: progress(n, l))
            x = library.export_rekordbox_xml(paths, name, export_root=Path(dest_root), copy=True)
            m["xml"] = x["xml"]
            return m

        self._run_async(_do, self._on_exported, "EXPORTING…")

    def _on_exported(self, res):
        msg = (f"Exported {res['copied']} tracks to:\n{res['dest']}\n\n"
               f"rekordbox XML (carries BPM/key/rating/colour/tags/cues):\n{res.get('xml','')}\n"
               f"  → in rekordbox: Preferences ▸ Advanced ▸ Database ▸ rekordbox xml, then drag\n"
               f"    the playlist out of the 'rekordbox xml' tree.\n\n"
               f"Simple playlist (files only):\n{res['m3u8']}\n"
               f"  → File ▸ Import Playlist ▸ pick that .m3u8.")
        if res["missing"]:
            msg += f"\n\n{len(res['missing'])} missing (skipped)."
        QMessageBox.information(self, "Export complete", msg)
        self.status.showMessage(f"EXPORTED {res['copied']} → {res['dest']}")

    # --- audio + selection ---
    def _selected_track(self):
        sel = self.table.selectionModel().selectedRows()
        return self.model.track_at(sel[0].row()) if sel else None

    def _table_menu(self, pos):
        """Right-click → reassign the selected track(s)' ARTIST(s) to a bucket (dj/personal). The
        virtual per-artist bucket: moves every track of that artist, no files touched, persists."""
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        tracks = [self.model.track_at(r) for r in rows]
        tracks = [t for t in tracks if t]
        if not tracks:
            return
        # distinct artists in the selection, keyed by the canonical filing-folder artist
        artists = {}
        for t in tracks:
            artists.setdefault(library.artist_key(t.path, t.artist), t)
        names = sorted({library._folder_artist(t.path) or library.primary_artist(t.artist)
                        for t in tracks})
        label = names[0] if len(names) == 1 else f"{len(names)} artists"
        targets = [b for b in ("dj", "personal") if b] + \
                  [b for b in library.list_buckets() if b not in ("dj", "personal")]
        menu = QMenu(self)
        for b in dict.fromkeys(["dj", "personal", *targets]):
            menu.addAction(f"Move {label} → {b.upper()}",
                           lambda _=False, bk=b: self._move_artists(artists, bk))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _move_artists(self, artists: dict, bucket: str):
        moved = 0
        for t in artists.values():
            moved += library.set_artist_bucket(t.artist, bucket, path=t.path)
        self._fill_bucket_box()
        self.refresh()
        self.status.showMessage(
            f"Moved {len(artists)} artist(s) → {bucket.upper()} ({moved} tracks)")

    def _on_table_select(self, *_):
        # selecting a row = INSPECT only; it never touches the player / NOW readout
        t = self._selected_track()
        if t:
            self.inspect(t)
            self.tag_drawer.set_track(t)     # deliberate selection drives the tag inspector

    def on_track_meta_changed(self, track):
        """A tag/colour/comment edit in the drawer changed `track` — refresh its table row."""
        tags = library.get_track_tags(track.path)
        vals = sorted({v for vs in tags.values() for v in vs}, key=str.lower)
        track.tag_summary = ", ".join(vals)
        r = self.model.row_for_path(track.path)
        if r is not None:
            self.model.refresh_row(r)

    def _track_line(self, tag, t):
        bpm = f"{t.bpm:.0f} BPM" if t.bpm else "—"
        rat = ("  ★" * t.rating) if t.rating else ""
        extra = ""
        if t.danceability is not None:
            extra += f" · dance {t.danceability:.2f}"
        if t.lufs is not None:
            extra += f" · {t.lufs:.1f} LUFS"
        return f"{tag}   {t.artist} — {t.title}   ·   {t.key or '—'} · {bpm}{extra}{rat}"

    def inspect(self, t, select=True):
        """The INSPECT channel: what you're looking at (selected row / hovered dot). Not the player.
        `select=False` = a transient readout (e.g. a 3D hover) that must NOT become the selection
        that ＋ADD / COMPATIBLE act on — only a deliberate click sets that."""
        if select:
            self.selected_track = t
        self.inspect_label.setText(self._track_line("◇ INSPECT", t))

    def select_track(self, t):
        """Map dot clicked: inspect it AND play it (a manual pick restarts the MAP walk)."""
        self.inspect(t)
        self.tag_drawer.set_track(t)   # a deliberate map pick drives the tag inspector too
        self.journey.clear()           # a manual pick is a new starting point for the nearest walk
        self.preview_track(t)

    def preview_track(self, t, navigating=False):
        """The PLAYING channel: load + play `t`. Updates only NOW (readout, row marker, map dot +
        trail) — never the INSPECT selection, so auto-advance can't hijack what you're looking at.
        `navigating=True` means this play came from prev/next walking the history cursor — so we
        don't re-log history, don't reset the back cursor, and don't add to the MAP walk set."""
        self.playing_track = t
        if not navigating:
            self._back = 0                 # a fresh play is a new head for the history walk
            self.journey.add(t.path)
        self.player.setSource(QUrl.fromLocalFile(t.path))
        self.player.play()
        self.now_label.setText(self._track_line("▶ NOW", t))
        self.model.set_playing(t.path)
        self._scroll_to_playing()
        self._load_waveform(t)
        if self.map_view is not None:
            self.map_view.set_playing(t.path)
        if self.map_view_3d is not None:
            self.map_view_3d.set_playing(t.path)
        if not navigating:
            self._log_history(t)
        trail = [x.path for x in reversed(self.history)][-40:]
        if self.map_view is not None:
            self.map_view.set_trail(trail)
        if self.map_view_3d is not None:
            self.map_view_3d.set_trail(trail)

    def _scroll_to_playing(self):
        if self.playing_track is None:
            return
        r = self.model.row_for_path(self.playing_track.path)
        if r is not None:
            self.table.scrollTo(self.model.index(r, 0), QAbstractItemView.PositionAtCenter)

    def _log_history(self, t):
        if self.history and self.history[0].path == t.path:
            return                                   # don't double-log a replay of the same track
        self.history = [x for x in self.history if x.path != t.path]
        self.history.insert(0, t)
        del self.history[200:]
        if getattr(self, "_hist", None) is not None and self._hist.isVisible():
            self._hist.refresh()

    def open_history(self):
        if not getattr(self, "_hist", None):
            self._hist = map_view.HistoryDialog(self, self)
        self._hist.refresh()
        self._hist.show()
        self._hist.raise_()

    def preview_selected(self):
        t = self._selected_track()
        if not t:
            self.status.showMessage("Select a track to preview.")
            return
        self.preview_track(t)

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        elif self.player.source().isEmpty():
            self.preview_selected()
        else:
            self.player.play()

    # --- playback queue (Phase C) ---
    def _set_shuffle(self, on):
        self.shuffle = bool(on)

    def _set_repeat(self, on):
        self.repeat = bool(on)

    def _on_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:   # current track finished -> advance
            self.play_next(auto=True)

    def _walk_view(self):
        """The map the MAP-walk should follow: the 3D galaxy when it's showing, else the 2D map."""
        if self.threed_toggle.isChecked() and self.map_view_3d is not None:
            return self.map_view_3d
        return self.map_view

    def play_next(self, auto=False):
        """LIST: next in the current sort (or a random track if shuffling). MAP: nearest
        unplayed dot (forward-only walk). `auto` distinguishes track-end from a button press."""
        walk = self._walk_view()
        if self.stack.currentIndex() == 1 and walk is not None:
            if self._back > 0:                          # stepped back earlier — walk forward again
                self._back -= 1
                self.preview_track(self.history[self._back], navigating=True)
                return
            cur = self.playing_track
            nxt = walk.nearest_unplayed(cur.path if cur else None, self.journey)
            if nxt is None:
                self.status.showMessage(
                    "MAP walk complete — nearby tracks all played. Click a dot to start a new walk.")
                return
            self.preview_track(nxt)
            return
        tracks = self.model.tracks
        if not tracks:
            return
        cur = self.playing_track
        if self.shuffle:
            pool = [t for t in tracks if not (cur and t.path == cur.path)] or tracks
            self.preview_track(random.choice(pool))
            return
        i = next((k for k, t in enumerate(tracks) if cur and t.path == cur.path), None)
        nxt = 0 if i is None else i + 1
        if nxt >= len(tracks):
            if not self.repeat:
                if auto:
                    self.status.showMessage("End of queue.")
                return
            nxt = 0
        self.preview_track(tracks[nxt])

    def play_prev(self):
        if self.stack.currentIndex() == 1:        # MAP: step back through the play history
            target = self._back + 1
            if target >= len(self.history):
                self.status.showMessage("No earlier track in history.")
                return
            self._back = target
            self.preview_track(self.history[target], navigating=True)
            return
        tracks = self.model.tracks
        if not tracks:
            return
        cur = self.playing_track
        i = next((k for k, t in enumerate(tracks) if cur and t.path == cur.path), None)
        if i is None:
            prv = 0
        elif i == 0:
            prv = len(tracks) - 1 if self.repeat else 0
        else:
            prv = i - 1
        self.preview_track(tracks[prv])

    def _on_play_state(self, state):
        # ▮▮ (Geometric Shapes, same block as ▶) renders as text — unlike ⏸ (U+23F8), which
        # falls back to a colour emoji on Windows (the "blue square" the others didn't have)
        self.play_btn.setText("▮▮" if state == QMediaPlayer.PlayingState else "▶")

    def _on_position(self, ms):
        self.waveform.set_position(ms)
        self.time_label.setText(f"{_fmt_ms(ms)} / {_fmt_ms(self.player.duration())}")

    def _on_duration(self, ms):
        self.waveform.set_duration(ms)

    # --- waveform (async load) + cue interaction (routed from WaveformWidget) ---
    def _load_waveform(self, track):
        """Load the track's cues + colored waveform into the strip. Reads the precomputed sidecar
        (cached snapshot → ~ms). The soundfile fallback for un-analyzed tracks can take ~0.6s; the
        share build's local analysis precomputes every waveform so that path is rarely hit."""
        self.waveform.set_cues(library.get_cues(track.path))
        dur = int((track.duration or 0) * 1000) or self.player.duration()
        self.waveform.set_waveform(library.get_waveform(track.path), dur)

    def _refresh_cues(self):
        if self.playing_track:
            self.waveform.set_cues(library.get_cues(self.playing_track.path))

    def on_waveform_seek(self, ms):
        self.player.setPosition(int(ms))

    def on_cue_jump(self, ms):
        self.player.setPosition(int(ms))

    def on_cue_delete(self, cue):
        library.delete_cue(cue["id"])
        self._refresh_cues()
        self.status.showMessage(f"Deleted {cue['kind']} cue {cue['idx']}")

    def on_cue_move(self, cue, ms):
        library.update_cue(cue["id"], ms)
        self._refresh_cues()

    def drop_memory_cue(self, ms=None):
        """Double-click on the waveform: drop a memory cue at `ms` (or the playhead) on the PLAYING track."""
        if not self.playing_track:
            self.status.showMessage("Play a track first, then drop a cue.")
            return
        at = int(self.player.position() if ms is None else ms)
        n = sum(1 for c in library.get_cues(self.playing_track.path) if c["kind"] == "memory")
        library.add_cue(self.playing_track.path, "memory", str(n + 1), at)
        self._refresh_cues()
        self.status.showMessage(f"Memory cue @ {_fmt_ms(at)}")

    def drop_hot_cue(self, n):
        """Keys 1-8 (with the waveform focused): set hot cue n at the playhead on the PLAYING track.
        Re-pressing the same number moves that hot cue to the current playhead."""
        if not self.playing_track:
            self.status.showMessage("Play a track first, then press 1-8 to set a hot cue.")
            return
        at = int(self.player.position())
        # one hot cue per slot: drop an existing hot cue with this index, then re-add at the playhead
        for c in library.get_cues(self.playing_track.path):
            if c["kind"] == "hot" and str(c["idx"]) == str(n):
                library.delete_cue(c["id"])
        library.add_cue(self.playing_track.path, "hot", str(n), at)
        self._refresh_cues()
        self.status.showMessage(f"Hot cue {n} @ {_fmt_ms(at)}  (rekordbox hot cue {chr(64 + n)})")

    def eventFilter(self, obj, event):
        # middle-mouse drag = jog/scrub the playing track (over the table or transport bar)
        et = event.type()
        if et == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
            self._jog = event.position().x()
            self._jog_base = self.player.position()
            return True
        if et == QEvent.MouseMove and self._jog is not None:
            dx = event.position().x() - self._jog
            dur = self.player.duration()
            self.player.setPosition(
                max(0, min(dur, int(self._jog_base + dx * map_view.JOG_MS_PER_PX))))
            return True
        if et == QEvent.MouseButtonRelease and event.button() == Qt.MiddleButton \
                and self._jog is not None:
            self._jog = None
            return True
        return super().eventFilter(obj, event)

    # --- rating (keys 0-5 on the table) ---
    def rate_selected(self, n):
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        for r in rows:
            t = self.model.track_at(r)
            library.set_rating(t.path, n)
            t.rating = n or None
            self.model.refresh_row(r)
        if rows:
            self.status.showMessage(
                f"Rated {len(rows)} track(s): {'★' * n if n else 'cleared'}")

    # --- delete / admin ---
    def do_delete(self):
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        if not rows:
            self.status.showMessage("Select tracks to delete.")
            return
        tracks = [self.model.track_at(r) for r in rows]
        names = "\n".join(f"• {t.artist} — {t.title}" for t in tracks[:12])
        more = "" if len(tracks) <= 12 else f"\n…and {len(tracks) - 12} more"
        ok = QMessageBox.question(
            self, "Move to trash",
            f"Move {len(tracks)} track(s) out of the library?\n\n{names}{more}\n\n"
            f"They move to the trash (reversible) — open TRASH to restore or delete for good.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ok != QMessageBox.Yes:
            return
        paths = [t.path for t in tracks]
        if not self.player.source().isEmpty() and self.player.source().toLocalFile() in paths:
            self.player.stop()
            self.player.setSource(QUrl())
        res = library.delete_tracks(paths)
        self.refresh()
        msg = f"Moved {res['moved']} to trash"
        if res["failed"]:
            msg += f" · {len(res['failed'])} failed"
        self.status.showMessage(msg)

    def _update_trash_actions(self, in_trash):
        """In the TRASH folder, swap the move-to-trash button for RESTORE / DELETE FOREVER."""
        self.restore_btn.setVisible(in_trash)
        self.purge_btn.setVisible(in_trash)
        self.delete_btn.setVisible(not in_trash)
        if getattr(self, "harmonic_btn", None):
            self.harmonic_btn.setEnabled(not in_trash)
        for b in ("chain_btn", "draft_btn", "arc_box"):       # set-building is meaningless in TRASH
            w = getattr(self, b, None)
            if w is not None:
                w.setEnabled(not in_trash)

    def _selected_trash_relpaths(self):
        """relpaths (under the trash root) for the selected rows in the TRASH view."""
        qroot = Path(library.QUARANTINE)
        rels = []
        for r in sorted({i.row() for i in self.table.selectionModel().selectedRows()}):
            t = self.model.track_at(r)
            try:
                rels.append(Path(t.path).relative_to(qroot).as_posix())
            except ValueError:
                continue
        return rels

    def restore_selected(self):
        rels = self._selected_trash_relpaths()
        if not rels:
            self.status.showMessage("Select trashed tracks to restore.")
            return
        if not self.player.source().isEmpty():     # stop if we're previewing one we're moving
            self.player.stop()
            self.player.setSource(QUrl())
        res = library.restore_tracks(rels)
        self.refresh()
        self.status.showMessage(f"Restored {res['restored']}"
                                + (f" · {len(res['failed'])} failed" if res["failed"] else ""))

    def delete_forever_selected(self):
        rels = self._selected_trash_relpaths()
        if not rels:
            self.status.showMessage("Select trashed tracks to delete for good.")
            return
        if QMessageBox.question(
                self, "Delete forever",
                f"Permanently delete {len(rels)} file(s) from disk? This cannot be undone.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if not self.player.source().isEmpty():
            self.player.stop()
            self.player.setSource(QUrl())
        res = library.purge_quarantine(rels)
        self.refresh()
        self.status.showMessage(f"Deleted {res['purged']} for good"
                                + (f" · {len(res['failed'])} failed" if res["failed"] else ""))


def _maybe_first_run():
    """On a fresh machine (no config yet), point Crate at the user's music folder so it isn't
    staring at the original author's Z:\\ drive. The rest (BPM/key/map/waveforms) comes from
    running analysis/analyze_all.py against the same folder."""
    if library.CONFIG_PATH.exists():
        return
    QMessageBox.information(
        None, "Welcome to Crate",
        "Let's point Crate at your music.\n\nPick the folder that holds your tracks (subfolders "
        "are fine). You can change this anytime via ⚙ FOLDERS.\n\nThen: click RE-INDEX to scan it, "
        "and ANALYZE to compute BPM/key, the sonic map, and waveforms (first ANALYZE downloads the "
        "embedding model once).")
    folder = QFileDialog.getExistingDirectory(None, "Select your music library folder")
    if not folder:
        return                                   # cancelled — they can set it later in ⚙ FOLDERS
    library.set_lib_root(folder)
    cfg = library.load_config()
    cfg["scan_roots"] = [{"label": Path(folder).name or "music", "path": folder}]
    library.save_config(cfg)


def main():
    app = QApplication(sys.argv)
    # taskbar groups by this id on Windows -> our icon shows instead of python's
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("crate.dj.prep")
    except Exception:
        pass
    icon = library.resource_dir() / "assets" / "crate_icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    theme.apply(app, library.get_skin())
    _maybe_first_run()
    win = CrateWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
