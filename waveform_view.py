"""waveform_view.py — Crate's colored DJ waveform + cue strip (replaces the plain seek slider).

Renders a 3-band (low/mid/high) waveform from the precomputed sidecar (library.get_waveform),
with a playhead, a brightened played region, and hot/memory cue flags drawn on the audio.

Interaction (routed back to the host window):
  click/drag empty   = seek/scrub        click a cue flag  = jump to it
  right-click a flag = delete the cue     drag a flag        = move the cue
Hot cues are dropped at the playhead via number keys 1-8 (handled in app.py).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QLinearGradient, QPolygonF
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QWidget

# band colours (rekordbox-ish): lows = blue, mids = amber, highs = near-white
C_LOW = np.array([70, 140, 255])
C_MID = np.array([255, 180, 70])
C_HIGH = np.array([235, 240, 255])

# named cue colours (match app.TRACK_COLORS); used for the flags
CUE_COLORS = {
    "pink": "#ff7ab8", "red": "#ff5b5b", "orange": "#ff9f43", "yellow": "#ffd93b",
    "green": "#5bd97a", "aqua": "#48d6d6", "blue": "#5c9dff", "purple": "#b58cff",
}
HOT_DEFAULT = "#ff9f43"      # orange
MEM_DEFAULT = "#48d6d6"      # aqua
HIT_PX = 7                   # click tolerance for grabbing a cue flag


class WaveformWidget(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.arr = None            # (N,3) uint8 or None
        self.dur = 0               # ms
        self.pos = 0               # ms
        self.cues = []             # list of dicts (from library.get_cues)
        self._drag_cue = None      # cue being dragged
        self._scrubbing = False
        self.view_start = 0.0      # zoom window: fraction of the track at the left edge
        self.view_span = 1.0       # ...and the fraction of the track that's visible (1 = whole)
        self.setMinimumHeight(72)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.ClickFocus)   # clicking the waveform focuses it -> hot-cue keys 1-8

    # --- data ---
    def set_waveform(self, arr, dur_ms):
        self.arr = arr
        self.dur = int(dur_ms or 0)
        self.view_start, self.view_span = 0.0, 1.0   # a new track resets the zoom to full
        self.update()

    def set_position(self, ms):
        self.pos = int(ms)
        self.update()

    def set_duration(self, ms):
        self.dur = int(ms or 0)
        self.update()

    def set_cues(self, cues):
        self.cues = list(cues or [])
        self.update()

    # --- geometry (respects the zoom window view_start/view_span) ---
    def _x_to_ms(self, x):
        w = max(1, self.width())
        frac = self.view_start + (x / w) * self.view_span
        return int(max(0, min(self.dur, frac * self.dur)))

    def _ms_to_x(self, ms):
        if self.dur <= 0:
            return 0
        frac = ms / self.dur
        return int((frac - self.view_start) / self.view_span * self.width())

    def _cue_at(self, x):
        for c in self.cues:
            if abs(self._ms_to_x(c["position_ms"]) - x) <= HIT_PX:
                return c
        return None

    # --- paint ---
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        mid_y = h / 2
        p.fillRect(self.rect(), QColor("#070810"))
        if self.arr is None or len(self.arr) == 0:
            p.setPen(QColor(244, 247, 251, 90))
            p.drawText(self.rect(), Qt.AlignCenter, "— no waveform (run the analysis pipeline) —")
            return

        n = len(self.arr)
        play_x = self._ms_to_x(self.pos)
        for x in range(w):
            frac = self.view_start + (x / w) * self.view_span    # zoom window -> bin
            i = int(frac * n)
            if i < 0 or i >= n:
                continue
            lo, md, hi = (float(v) for v in self.arr[min(i, n - 1)])
            s = lo + md + hi
            if s <= 0:
                continue
            amp = max(lo, md, hi) / 255.0
            col = (C_LOW * lo + C_MID * md + C_HIGH * hi) / s     # band-weighted blend
            played = x <= play_x
            a = 255 if played else 90                              # dim the unplayed region
            p.setPen(QColor(int(col[0]), int(col[1]), int(col[2]), a))
            half = amp * (mid_y - 2)
            p.drawLine(x, int(mid_y - half), x, int(mid_y + half))

        # centre baseline
        p.setPen(QColor(244, 247, 251, 22))
        p.drawLine(0, int(mid_y), w, int(mid_y))

        # cue flags
        for c in self.cues:
            self._draw_cue(p, c, h)

        # playhead
        p.setPen(QPen(QColor(255, 255, 255, 230), 1))
        p.drawLine(play_x, 0, play_x, h)

    def _draw_cue(self, p, c, h):
        x = self._ms_to_x(c["position_ms"])
        name = (c.get("color") or "").lower()
        hexc = CUE_COLORS.get(name) or (HOT_DEFAULT if c["kind"] == "hot" else MEM_DEFAULT)
        col = QColor(hexc)
        p.setPen(QPen(col, 1))
        p.drawLine(x, 0, x, h)
        # a little flag at the top with the cue index
        if c["kind"] == "hot":
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            tri = QPolygonF([QPointF(x, 0), QPointF(x + 11, 0), QPointF(x, 11)])
            p.drawPolygon(tri)
            p.setPen(QColor("#070810"))
            p.drawText(x + 1, 9, str(c["idx"])[:1])
        else:
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            p.drawRect(x - 1, 0, 3, 7)

    # --- interaction ---
    def mousePressEvent(self, e):
        x = int(e.position().x())
        c = self._cue_at(x)
        if e.button() == Qt.RightButton:
            if c:
                self.host.on_cue_delete(c)
            return
        if e.button() == Qt.LeftButton:
            if c:
                self._drag_cue = c
                self.host.on_cue_jump(c["position_ms"])
                return
            self._scrubbing = True
            self.host.on_waveform_seek(self._x_to_ms(x))

    def mouseMoveEvent(self, e):
        x = int(e.position().x())
        if self._drag_cue is not None:
            self._drag_cue["position_ms"] = self._x_to_ms(x)   # live preview
            self.update()
            return
        if self._scrubbing:
            self.host.on_waveform_seek(self._x_to_ms(x))
            return
        self.setCursor(Qt.SizeHorCursor if self._cue_at(x) else Qt.PointingHandCursor)

    def mouseReleaseEvent(self, e):
        if self._drag_cue is not None:
            self.host.on_cue_move(self._drag_cue, self._x_to_ms(int(e.position().x())))
            self._drag_cue = None
            return
        self._scrubbing = False

    def mouseDoubleClickEvent(self, e):
        # double-click empty waveform = drop a memory cue at THAT point (no modal)
        x = int(e.position().x())
        if e.button() == Qt.LeftButton and self._cue_at(x) is None:
            self.host.drop_memory_cue(self._x_to_ms(x))

    def wheelEvent(self, e):
        # shift + scroll = zoom in/out, anchored on the spot under the cursor
        if not (e.modifiers() & Qt.ShiftModifier) or self.dur <= 0:
            e.ignore()
            return
        w = max(1, self.width())
        x = e.position().x()
        anchor = self.view_start + (x / w) * self.view_span          # track-fraction under cursor
        factor = 1 / 1.25 if e.angleDelta().y() > 0 else 1.25        # scroll up = zoom in
        span = min(1.0, max(0.01, self.view_span * factor))
        start = min(max(0.0, anchor - (x / w) * span), 1.0 - span)   # keep `anchor` under the cursor
        self.view_start, self.view_span = start, span
        self.update()
        e.accept()
