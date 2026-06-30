"""tag_drawer.py — Crate's inline TRACK inspector (replaces the popup tag modal).

A collapsible bottom drawer that edits the SELECTED track's DJ metadata in place: rekordbox
My-Tag category chips (click to toggle) + free tags + colour swatches + a comment. Everything
auto-saves on change — no Save button, no modal. Cues live on the waveform (see waveform_view).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QSize, QRect, QPoint
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QLayout,
    QSizePolicy, QCompleter,
)

import library

# imported lazily-ish to avoid a circular import at module load
TAG_CATEGORIES = ["genre", "components", "vocal", "situation", "mood"]
TRACK_COLORS = ["(none)", "pink", "red", "orange", "yellow", "green", "aqua", "blue", "purple"]
TRACK_COLOR_HEX = {
    "pink": "#ff7ab8", "red": "#ff5b5b", "orange": "#ff9f43", "yellow": "#ffd93b",
    "green": "#5bd97a", "aqua": "#48d6d6", "blue": "#5c9dff", "purple": "#b58cff",
}


class FlowLayout(QLayout):
    """Left-to-right wrapping layout (Qt's flow-layout example) for chips."""

    def __init__(self, spacing=6):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._do(QRect(0, 0, w, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        return s + QSize(2, 2)

    def _do(self, rect, test):
        x, y, line_h = rect.x(), rect.y(), 0
        sp = self.spacing()
        for it in self._items:
            sh = it.sizeHint()
            if x + sh.width() > rect.right() and line_h > 0:
                x = rect.x()
                y += line_h + sp
                line_h = 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), sh))
            x += sh.width() + sp
            line_h = max(line_h, sh.height())
        return y + line_h - rect.y()


class TagDrawer(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.track = None
        self.setObjectName("tagDrawer")
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 4, 2, 2)
        v.setSpacing(4)

        # header is one big clickable bar that collapses/expands the drawer (also: Ctrl+T)
        self._desc = "— select a track —"
        self.header_btn = QPushButton()
        self.header_btn.setObjectName("drawerHeader")
        self.header_btn.setCursor(Qt.PointingHandCursor)
        self.header_btn.setToolTip("Click (or Ctrl+T) to collapse / expand the tag inspector")
        self.header_btn.clicked.connect(self.toggle_collapsed)
        v.addWidget(self.header_btn)

        self.body = QWidget()
        bl = QVBoxLayout(self.body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(3)
        # one flow-row of chips per category (+ a small add field)
        self.cat_flows = {}
        for cat in TAG_CATEGORIES + ["free"]:
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = QLabel(("FREE" if cat == "free" else cat.upper()).ljust(10))
            lbl.setObjectName("readout")
            lbl.setFixedWidth(86)
            lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            chips = QWidget()
            flow = FlowLayout()
            chips.setLayout(flow)
            add = QLineEdit(placeholderText="+ add")
            add.setFixedWidth(84)
            add.returnPressed.connect(lambda c=cat: self._add_new(c))
            self.cat_flows[cat] = (flow, chips, add)
            row.addWidget(lbl)
            row.addWidget(add)            # add field on the LEFT, next to the label
            row.addWidget(chips, 1)
            bl.addLayout(row)

        # colour swatches + comment
        crow = QHBoxLayout()
        crow.setSpacing(6)
        clbl = QLabel("COLOUR".ljust(10))
        clbl.setObjectName("readout")
        clbl.setFixedWidth(86)
        crow.addWidget(clbl)
        self.swatches = {}
        for name in TRACK_COLORS:
            b = QPushButton("✕" if name == "(none)" else "")
            b.setFixedSize(20, 20)
            b.setCheckable(True)
            b.setToolTip(name)
            if name != "(none)":
                b.setStyleSheet(f"background:{TRACK_COLOR_HEX[name]}; border-radius:10px;")
            b.clicked.connect(lambda _=False, n=name: self._set_color(n))
            self.swatches[name] = b
            crow.addWidget(b)
        crow.addStretch(1)
        clbl2 = QLabel("COMMENT")
        clbl2.setObjectName("readout")
        crow.addWidget(clbl2)
        self.comment = QLineEdit(placeholderText="free comment…")
        self.comment.editingFinished.connect(self._set_comment)
        crow.addWidget(self.comment, 2)
        bl.addLayout(crow)
        v.addWidget(self.body)
        self._refresh_header()
        self.set_track(None)

    # --- collapse ---
    def _refresh_header(self):
        arrow = "▾" if self.body.isVisible() else "▸"
        self.header_btn.setText(f"  {arrow}   TRACK   ◇   {self._desc}")

    def toggle_collapsed(self):
        self.body.setVisible(not self.body.isVisible())
        self._refresh_header()

    # --- populate for a track ---
    def set_track(self, track):
        self.track = track
        on = track is not None
        self.body.setEnabled(on)
        if not on:
            self._desc = "— select a track —"
            self._refresh_header()
            return
        self._desc = f"{track.artist} — {track.title}"
        self._refresh_header()
        existing = library.get_track_tags(track.path)
        for cat in TAG_CATEGORIES + ["free"]:
            flow, chips, add = self.cat_flows[cat]
            self._clear(flow)
            active = set(existing.get(cat, []))
            values = sorted(set(library.all_tag_values(cat)) | active, key=str.lower)
            for val in values:
                flow.addWidget(self._chip(cat, val, val in active))
            add.setCompleter(QCompleter(library.all_tag_values(cat)))
            chips.adjustSize()
            chips.updateGeometry()
        # colour
        for name, b in self.swatches.items():
            b.setChecked((track.color or "(none)") == name)
        self.comment.blockSignals(True)
        self.comment.setText(track.comment or "")
        self.comment.blockSignals(False)

    def _clear(self, flow):
        while flow.count():
            it = flow.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

    def _chip(self, cat, value, active):
        b = QPushButton(value)
        b.setObjectName("chip")
        b.setCheckable(True)
        b.setChecked(active)
        b.setCursor(Qt.PointingHandCursor)
        b.toggled.connect(lambda on, c=cat, v=value: self._toggle(c, v, on))
        return b

    # --- edits (auto-save) ---
    def _toggle(self, cat, value, on):
        if not self.track:
            return
        if on:
            library.add_track_tag(self.track.path, cat, value)
        else:
            library.remove_track_tag(self.track.path, cat, value)
        self.host.on_track_meta_changed(self.track)

    def _add_new(self, cat):
        flow, chips, add = self.cat_flows[cat]
        val = add.text().strip()
        if not val or not self.track:
            return
        library.add_track_tag(self.track.path, cat, val)
        add.clear()
        # add a checked chip if it isn't already shown
        for i in range(flow.count()):
            w = flow.itemAt(i).widget()
            if w and w.text().lower() == val.lower():
                w.setChecked(True)
                self.host.on_track_meta_changed(self.track)
                return
        flow.addWidget(self._chip(cat, val, True))
        chips.adjustSize()
        self.host.on_track_meta_changed(self.track)

    def _set_color(self, name):
        if not self.track:
            return
        for n, b in self.swatches.items():
            b.setChecked(n == name)
        color = None if name == "(none)" else name
        library.set_color(self.track.path, color)
        self.track.color = color
        self.host.on_track_meta_changed(self.track)

    def _set_comment(self):
        if not self.track:
            return
        c = self.comment.text().strip() or None
        library.set_comment(self.track.path, c)
        self.track.comment = c
        self.host.on_track_meta_changed(self.track)
