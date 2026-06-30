"""map_view.py — Crate's UMAP music map (plexus + force re-layout + export).

Dots = tracks, positioned by CLAP-embedding UMAP, colored (outline) by Camelot key, sized by
rating. Pick a CONNECT relation (sonic/key/tempo/artist) to draw a plexus AND physically
re-arrange the points (force-directed) so that relationship becomes the spatial structure.

This is an embeddable widget (`MapView`) — the host window (app.py) owns the transport, crate
panel, and history; the map just renders dots/edges and routes interaction back to the host.

Interaction:
  scroll = zoom    drag empty space = pan    click a dot = select + play (host bottom bar)
  middle-drag left/right = jog/scrub the playing track
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, QLineF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsTextItem, QVBoxLayout, QLabel, QDialog, QListWidget, QListWidgetItem,
)

JOG_MS_PER_PX = 45      # middle-mouse horizontal drag -> seek sensitivity

import library


def _fmt_ms(ms):
    s = max(0, int(ms) // 1000)
    return f"{s // 60}:{s % 60:02d}"

W, H, M = 1500, 1000, 60       # scene extent + margin
S = 10.0                        # internal sim-space scale
R = 6                           # base dot radius
KNN = 12
KNN_OVERSAMPLE = 4     # take this many × KNN top cosine candidates, then re-rank by mixability
KNN_PENALTY_W = 0.5    # how hard tempo/key incompatibility demotes a sonic neighbour (0 = pure sonic)
ASPECT = (H - 2 * M) / (W - 2 * M)   # 0.638 — multiply an x-extent by this so a circle in sim
                                     # space draws as a circle on the (wider) screen, not an ellipse

DENSITY = {"Sparse": 2, "Medium": 4}   # edges per node (Dense removed — too many lines, slow)
EDGE_COLOR = {
    "sonic":  QColor(180, 210, 255, 55),
    "key":    QColor(150, 230, 180, 70),
    "tempo":  QColor(240, 190, 120, 70),
    "artist": QColor(225, 150, 230, 80),
}


def _cluster_color(cid) -> QColor:
    """Stable, well-separated colour per full-d HDBSCAN cluster id; noise/unclustered (-1/None) = grey.
    Golden-ratio hue walk so any number of clusters stays visually distinct."""
    if cid is None or cid < 0:
        return QColor(110, 115, 125)
    return QColor.fromHsvF((cid * 0.61803398875) % 1.0, 0.6, 1.0)


def key_color(key: str | None) -> QColor:
    p = library._parse_camelot(key) if key else None
    if not p:
        return QColor(150, 150, 160)
    num, letter = p
    return QColor.fromHsvF((num - 1) / 12.0, 0.65, 1.0 if letter == "B" else 0.8)


def _tempo_color(bpm) -> QColor:
    """Continuous tempo ramp: slow = cool blue, fast = warm red. Unknown = grey."""
    if not bpm:
        return QColor(120, 120, 130)
    f = (min(max(float(bpm), 90.0), 170.0) - 90.0) / 80.0
    return QColor.fromHsvF(0.66 * (1 - f), 0.72, 1.0)


def _energy_color(energy) -> QColor:
    """Energy ramp: calm = teal, hot = orange. Unknown = grey. (energy ~0..0.25)"""
    if not energy:
        return QColor(120, 120, 130)
    f = min(max(float(energy) / 0.25, 0.0), 1.0)
    return QColor.fromHsvF(0.5 * (1 - f) + 0.08 * f, 0.7, 1.0)


def _dance_color(dance) -> QColor:
    """Danceability ramp: chill = cool blue, danceable = hot pink. Unknown = grey. (0..1)"""
    if dance is None:
        return QColor(120, 120, 130)
    f = min(max(float(dance), 0.0), 1.0)
    return QColor.fromHsvF(0.6 + 0.32 * f, 0.72, 1.0)


def _primary_artist(name: str) -> str:
    for sep in (";", "/", " feat", " ft", ","):
        if sep in name.lower():
            return name.lower().split(sep)[0].strip()
    return name.lower().strip()


def rich_info(t) -> str:
    bpm = f"{t.bpm:.0f} BPM" if t.bpm else "— BPM"
    rat = ("★" * t.rating) if t.rating else "unrated"
    dur = f"{int(t.duration)//60}:{int(t.duration)%60:02d}" if t.duration else "—"
    enbar = ""
    if t.energy:
        lvl = max(1, min(5, int(t.energy / 0.05) + 1))
        enbar = "▇" * lvl + "·" * (5 - lvl)
    lines = [f"{t.artist} — {t.title}", t.album or "",
             f"{t.key or '—'}    {bpm}    {rat}",
             f"energy {enbar or '—'}    {t.bucket} · {t.ext.lstrip('.')} · {dur}"]
    return "\n".join(l for l in lines if l)


class MapView(QGraphicsView):
    def __init__(self, tracks_xy, host):
        super().__init__()
        self.host = host
        self.tracks = [t for t, x, y in tracks_xy]
        self.base = np.array([[x, y] for t, x, y in tracks_xy], dtype=float)  # [0,1]
        self.pos = self.base.copy() * S
        self.mode, self.density = "none", DENSITY["Sparse"]
        self.edges = []
        self._edge_arr = None
        self._edge_lines = []        # cached QLineF (scene coords) painted in one drawBackground pass
        self._tweening = False       # edges are hidden mid-animation, repainted on settle
        self._hover = None
        self._jog = None
        self._subset = None        # None = show all; else a set of visible track paths
        self.playing_path = None   # emphasized "now playing" dot
        self.trail_paths = []      # listening trail, oldest -> newest (host pushes from history)
        self.trail_items = []
        self.color_mode = "key"    # COLOR BY: key | artist | tempo | energy | cluster
        self.cluster_labels = []   # artist-name labels over each cluster (CONNECT=Artist only)
        try:
            self.clusters = library.load_clusters()   # {path: full-d HDBSCAN cluster id} for COLOR BY=cluster
        except Exception:
            self.clusters = {}

        self.scene = QGraphicsScene(0, 0, W, H)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor("#05060a"))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMouseTracking(True)

        self.dot_items = []
        for i, t in enumerate(self.tracks):
            r = R + (t.rating or 0) * 0.7
            it = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            it.setZValue(1)
            it.setData(0, i)
            it.setToolTip(rich_info(t))
            self.scene.addItem(it)
            self.dot_items.append(it)
        self._restyle_all()

        # KNN for the sonic plexus. Prefer the TRUE (mean-centered) 512-d CLAP vectors so
        # "connect similar tracks" means sonically similar — not merely adjacent in the 2D
        # projection (a projection of a projection). Falls back to 2D coords for any track
        # without a loaded vector, or entirely if vectors aren't available.
        self._knn = self._build_knn()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tween_step)
        self._place_all()

    def _knn_2d(self, i):
        d = ((self.base - self.base[i]) ** 2).sum(1)
        return [j for j in np.argsort(d)[1:KNN + 1]]

    def _build_knn(self):
        """KNN per track from the real centered 512-d vectors, then RE-RANKED by DJ mixability so the
        sonic plexus connects tracks that actually MIX (key+tempo), not merely share a timbre — a 110
        house track and a 160 footwork track built on the same 808s shouldn't be wired together. Falls
        back to 2D coords for any track without a loaded vector."""
        try:
            vecs = library.load_vectors()
        except Exception:
            vecs = {}
        paths = [t.path for t in self.tracks]
        have = [i for i, p in enumerate(paths) if p in vecs]
        if len(have) < max(3, 0.5 * len(paths)):
            return [self._knn_2d(i) for i in range(len(self.base))]   # too few vectors -> 2D
        M = np.stack([vecs[paths[i]] for i in have])                  # unit rows (centered)
        sims = M @ M.T
        row_of = {i: a for a, i in enumerate(have)}
        knn = [None] * len(self.tracks)
        for a, i in enumerate(have):
            order = [have[b] for b in np.argsort(-sims[a]) if have[b] != i]
            cand = order[:KNN * KNN_OVERSAMPLE]                       # top cosine, then fuse-rerank
            ta = self.tracks[i]
            cand.sort(key=lambda j: -(float(sims[a][row_of[j]])
                                      - KNN_PENALTY_W * library.compat_penalty(ta, self.tracks[j])))
            knn[i] = cand[:KNN]
        for i in range(len(self.tracks)):
            if not knn[i]:                                            # no vector -> 2D neighbours
                knn[i] = self._knn_2d(i)
        return knn

    # --- styling ---
    def _dot_color(self, t) -> QColor:
        """The dot's base colour under the current COLOR BY mode."""
        m = self.color_mode
        if m == "artist":
            return _artist_color(_primary_artist(t.artist))
        if m == "tempo":
            return _tempo_color(t.bpm)
        if m == "energy":
            return _energy_color(t.energy)
        if m == "danceability":
            return _dance_color(getattr(t, "danceability", None))
        if m == "cluster":
            return _cluster_color(self.clusters.get(t.path))
        return key_color(t.key)

    def set_color_mode(self, mode):
        self.color_mode = mode
        self._restyle_all()

    def _restyle_dot(self, i):
        t = self.tracks[i]
        it = self.dot_items[i]
        c = self._dot_color(t)
        if t.path == self.playing_path:             # now playing -> bright fill + white halo, on top
            it.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 255)))
            pen = QPen(QColor(255, 255, 255, 255))
            pen.setWidthF(3.2)
            it.setPen(pen)
            it.setZValue(4)
            return
        it.setZValue(1)
        if t.path in self.host.crate_paths:        # in crate -> filled + white ring
            it.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 210)))
            pen = QPen(QColor(255, 255, 255, 230))
            pen.setWidthF(2.0)
        else:                                       # otherwise -> hollow outline
            it.setBrush(Qt.NoBrush)
            pen = QPen(c)
            pen.setWidthF(1.7)
        it.setPen(pen)

    def _clear_cluster_labels(self):
        for it in self.cluster_labels:
            self.scene.removeItem(it)
        self.cluster_labels = []

    def _show_cluster_labels(self, pos):
        """In CONNECT=Artist, label each artist's clump with the artist name, placed just above the
        cluster (or below if it's near the top edge) so names don't sit on top of the dots/clicks."""
        self._clear_cluster_labels()
        groups = defaultdict(list)
        for i, t in enumerate(self.tracks):
            if self.dot_items[i].isVisible():
                groups[_primary_artist(t.artist)].append(i)
        for art, mem in groups.items():
            if not art:
                continue
            pts = [self._to_scene(pos[i]) for i in mem]
            cx = sum(p[0] for p in pts) / len(pts)
            top = min(p[1] for p in pts)
            bot = max(p[1] for p in pts)
            lbl = QGraphicsTextItem(art)
            lbl.setDefaultTextColor(QColor(210, 215, 230))
            lbl.setAcceptedMouseButtons(Qt.NoButton)   # never intercept a click on the dots below
            f = lbl.font()
            f.setPointSizeF(7.0)
            lbl.setFont(f)
            lbl.setZValue(6)
            br = lbl.boundingRect()
            # above the cluster by default; flip below if it would clip off the top of the scene
            y = top - 16 if top - 16 > 8 else bot + 6
            lbl.setPos(cx - br.width() / 2, y)
            self.scene.addItem(lbl)
            self.cluster_labels.append(lbl)

    def _index_of(self, path):
        if path is None:
            return None
        return next((k for k, t in enumerate(self.tracks) if t.path == path), None)

    def set_playing(self, path):
        """Emphasize the now-playing dot (restyle old + new)."""
        old = self.playing_path
        self.playing_path = path
        for p in (old, path):
            i = self._index_of(p)
            if i is not None:
                self._restyle_dot(i)

    def set_trail(self, paths):
        """Draw the listening trail: a comet line threading the played dots in order, brightening
        toward the most recent. `paths` is oldest -> newest (host passes from its history)."""
        self.trail_paths = [p for p in paths if self._index_of(p) is not None]
        for it in self.trail_items:
            self.scene.removeItem(it)
        self.trail_items = []
        n = len(self.trail_paths)
        for s in range(n - 1):
            frac = (s + 1) / max(1, n - 1)           # 0..1, brighter/thicker toward newest
            ln = QGraphicsLineItem()
            pen = QPen(QColor(180, 210, 255, int(35 + 165 * frac)))
            pen.setWidthF(1.0 + 1.8 * frac)
            ln.setPen(pen)
            ln.setZValue(2)
            self.scene.addItem(ln)
            self.trail_items.append(ln)
        self._place_trail()

    def _restyle_all(self):
        for i in range(len(self.dot_items)):
            self._restyle_dot(i)

    # --- positions ---
    def _to_scene(self, p):
        return (M + (p[0] / S) * (W - 2 * M), M + (p[1] / S) * (H - 2 * M))

    def _place_dots(self):
        for i, it in enumerate(self.dot_items):
            x, y = self._to_scene(self.pos[i])
            it.setPos(x, y)

    def _scene_pts(self):
        """All dot positions in scene coords, vectorized (for edge painting)."""
        P = self.pos
        return (M + (P[:, 0] / S) * (W - 2 * M), M + (P[:, 1] / S) * (H - 2 * M))

    def _rebuild_edge_lines(self):
        """Recompute the cached QLineF list (scene coords) for the plexus. Cheap; called when
        positions / subset / mode change — NOT on pan/zoom (those just reuse the cache + Qt's view
        transform). Edges to hidden (subset-filtered) dots are skipped."""
        self._edge_lines = []
        if self._edge_arr is None or self.mode == "none":
            self.viewport().update()
            return
        xs, ys = self._scene_pts()
        if self._subset is None:
            vis = None
        else:
            vis = np.fromiter((t.path in self._subset for t in self.tracks), bool, len(self.tracks))
        lines = []
        for i, j in self._edge_arr:
            if vis is not None and (not vis[i] or not vis[j]):
                continue
            lines.append(QLineF(xs[i], ys[i], xs[j], ys[j]))
        self._edge_lines = lines
        self.viewport().update()

    def drawBackground(self, painter, rect):
        # one painter pass for the WHOLE plexus (was thousands of QGraphicsLineItems → slow scene
        # indexing + per-item paint). Edges sit behind the dots; cosmetic pen = constant width at
        # any zoom. Hidden during the artist-anchor tween for a clean slide.
        super().drawBackground(painter, rect)
        if self._tweening or not self._edge_lines:
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(EDGE_COLOR.get(self.mode, QColor(180, 210, 255, 55)))
        pen.setWidthF(0.9)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLines(self._edge_lines)

    def _place_trail(self):
        for s, ln in enumerate(self.trail_items):
            i = self._index_of(self.trail_paths[s])
            j = self._index_of(self.trail_paths[s + 1])
            if i is None or j is None:
                continue
            x1, y1 = self._to_scene(self.pos[i])
            x2, y2 = self._to_scene(self.pos[j])
            ln.setLine(x1, y1, x2, y2)

    def _place_all(self):
        self._place_dots()
        self._rebuild_edge_lines()
        self._place_trail()

    # --- connections + re-layout ---
    def set_mode(self, mode, k):
        """Change the CONNECT relationship (and DENSITY = edges per node). The chosen relationship
        becomes the SPATIAL structure via a DIRECT, legible layout (a force solve seeded from the
        UMAP never actually organizes at this library size — keys stayed un-grouped, tempo un-
        ordered). Each is computed in O(n) then eased in with one tween:
          • Sonic / None — the raw UMAP (already the sonic map).
          • Key   — every track placed on its KEY's spot on a Camelot wheel → all same-key tracks
                    clump together, compatible keys sit adjacent (the readable "key gradient").
          • Tempo — tracks sorted by BPM and snaked into ONE meandering line (position == tempo).
          • Artist— re-projected onto the artist-UMAP shape (similar artists near each other).
        """
        self.mode, self.density = mode, k
        self._clear_cluster_labels()
        self.timer.stop()
        self._build_edges()
        start = self.base.copy() * S                   # always re-seed from the UMAP
        if mode in ("none", "sonic"):
            self.pos = self._separate(start, shrink=0.94)   # de-overlap the raw UMAP, tighten a touch
            self._place_all()
            return
        if mode == "key":
            self._tween(start, self._key_target())
            return
        if mode == "tempo":
            self._tween(start, self._tempo_target())
            return
        if mode == "artist":
            anchored = self._artist_anchored_target()
            if anchored is not None:
                anchored = self._separate(anchored, shrink=0.95)
                self._show_cluster_labels(anchored)
                self._tween(start, anchored)
                return
        self.pos = start                               # fallback: leave at UMAP
        self._place_all()

    def _separate(self, P, min_px=12.0, iters=50, shrink=1.0):
        """Anti-overlap: push any two dots closer than `min_px` ON SCREEN apart until they clear,
        then (if shrink<1) pull the whole cloud toward its centre so the arrangement reads a little
        tighter/smaller. Runs in pixel space (uniform metric regardless of the x/y aspect) and maps
        back to sim space. O(n) per pass via a uniform grid (only neighbouring cells compared), so
        it's cheap enough to run inline on every mode switch. Keeps the layout's STRUCTURE — it only
        spreads coincident points so 'really similar' tracks stop stacking into one blob."""
        sx = (W - 2 * M) / S                           # sim-unit -> pixel (x)
        sy = (H - 2 * M) / S                           # sim-unit -> pixel (y)
        Q = np.column_stack([P[:, 0] * sx, P[:, 1] * sy]).astype(float)
        n = len(Q)
        rng = np.random.default_rng(7)
        cell = min_px
        md2 = min_px * min_px
        for _ in range(iters):
            disp = np.zeros_like(Q)
            cidx = np.floor(Q / cell).astype(np.int64)
            grid = {}
            for i in range(n):
                grid.setdefault((cidx[i, 0], cidx[i, 1]), []).append(i)
            any_push = False
            for (cx, cy), members in grid.items():
                neigh = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        g = grid.get((cx + dx, cy + dy))
                        if g:
                            neigh.extend(g)
                for i in members:
                    qix, qiy = Q[i]
                    for j in neigh:
                        if j <= i:
                            continue
                        ddx = qix - Q[j, 0]
                        ddy = qiy - Q[j, 1]
                        d2 = ddx * ddx + ddy * ddy
                        if d2 >= md2:
                            continue
                        if d2 < 1e-9:                   # exactly coincident -> random unit nudge
                            a = rng.random() * 6.283185
                            ux, uy, d = math.cos(a), math.sin(a), 0.0
                        else:
                            d = math.sqrt(d2)
                            ux, uy = ddx / d, ddy / d
                        push = (min_px - d) * 0.7
                        disp[i, 0] += ux * push; disp[i, 1] += uy * push
                        disp[j, 0] -= ux * push; disp[j, 1] -= uy * push
                        any_push = True
            Q += disp
            if not any_push:
                break
        if shrink != 1.0:
            c = Q.mean(0)
            Q = c + (Q - c) * shrink
        return np.column_stack([Q[:, 0] / sx, Q[:, 1] / sy])

    def _key_target(self):
        """Camelot-wheel layout: each track sits at its key's wheel position (12 o'clock = 1, going
        clockwise; inner ring = A/minor, outer = B/major). Same-key tracks share a spot, then the
        anti-overlap pass spreads them into a tight readable clump; adjacent wheel positions are
        harmonically compatible, so the whole map becomes a ring of key clusters. Unknown-key tracks
        park at the top edge. ASPECT keeps the wheel circular on screen; radii are modest so the
        whole cluster stays compact."""
        cx = cy = S / 2.0
        rng = np.random.default_rng(42)
        P = self.base.copy() * S
        # rings pushed apart + bigger so the 12 minor (inner) and 12 major (outer) clumps each have
        # enough circumference to read as SEPARATE keys instead of merging into one donut.
        r_in, r_out = 0.27 * S, 0.46 * S               # inner=minor, outer=major
        for i, t in enumerate(self.tracks):
            p = library._parse_camelot(t.key) if t.key else None
            if not p:
                P[i] = [0.5 * S, 0.03 * S]
                continue
            num, letter = p
            ang = (num - 1) / 12.0 * 2 * math.pi - math.pi / 2   # 1 at top, clockwise
            rad = r_in if letter == "A" else r_out
            jx, jy = rng.normal(0, 0.008, 2) * S
            P[i] = [cx + (rad * math.cos(ang) + jx) * ASPECT, cy + rad * math.sin(ang) + jy]
        return self._separate(P, shrink=0.95)

    def _tempo_target(self):
        """Sort tracks by BPM and lay them along ONE long flowing line — a gentle S-curve / strange-
        attractor meander that changes direction a few times (NOT a spiral). The line sweeps left
        (slowest) -> right (fastest) so position along it == tempo; the vertical flow is the sum of
        two out-of-phase sines so it reads as an organic, elegant curve rather than a mechanical
        wave. The anti-overlap pass thickens the line only where many tracks share a BPM (common
        tempos), keeping it a clean ribbon elsewhere. Untempo'd tracks park along the bottom."""
        P = self.base.copy() * S
        order = [i for i in sorted(range(len(self.tracks)),
                                   key=lambda x: self.tracks[x].bpm or 1e9) if self.tracks[i].bpm]
        n = len(order)
        x0, x1 = 0.08 * S, 0.92 * S                    # slowest left -> fastest right (position == BPM)
        cy = 0.50 * S
        a1, f1, ph1 = 0.24 * S, 1.25, 0.0              # primary graceful bend (~2-3 direction changes)
        a2, f2, ph2 = 0.07 * S, 2.6, 1.3               # secondary wave -> organic, attractor-like flow
        for rank, i in enumerate(order):
            f = rank / max(1, n - 1)
            y = cy + a1 * math.sin(f * 2 * math.pi * f1 + ph1) + a2 * math.sin(f * 2 * math.pi * f2 + ph2)
            P[i] = [x0 + f * (x1 - x0), y]
        for i, t in enumerate(self.tracks):
            if not t.bpm:
                P[i] = [0.5 * S, 0.97 * S]
        return self._separate(P, shrink=0.97)

    def _artist_coords(self) -> dict:
        """Lazy {artist_lower: (x, y)} from the artist-UMAP sidecar (same data as the ARTIST map)."""
        if getattr(self, "_acoords", None) is None:
            try:
                self._acoords = {a.lower(): (x, y) for a, x, y, n in library.artists_with_coords()}
            except Exception:
                self._acoords = {}
        return self._acoords

    def _artist_anchored_target(self):
        """Target positions that place each track on its artist's artist-UMAP coordinate (with a
        small jitter so same-artist tracks read as a clump). Tracks with no artist coord are
        gathered into one 'unmatched' clump in the bottom-left corner instead of being left
        stranded with long edges. Returns None if no coords / too few match (force-layout fallback)."""
        coords = self._artist_coords()
        if not coords:
            return None
        target = self.pos.copy()
        rng = np.random.default_rng(42)
        hits = 0
        for i, t in enumerate(self.tracks):
            a = _primary_artist(t.artist).lower()
            xy = coords.get(a)
            if xy is None:
                jx, jy = rng.normal(0, 0.02, 2)         # unmatched -> their own corner clump
                target[i] = [(0.04 + jx) * S, (0.96 + jy) * S]
                continue
            jx, jy = rng.normal(0, 0.018, 2)            # ~clump radius in [0,1] UMAP space
            target[i] = [(xy[0] + jx) * S, (xy[1] + jy) * S]
            hits += 1
        if hits < max(3, len(self.tracks) * 0.25):      # not enough matched -> not worth the remap
            return None
        return target

    def _tween(self, a, b, dur=0.85):
        # hide the plexus + trail + labels during the slide so only the dots move -> smooth.
        self._tweening = True
        for it in self.trail_items + self.cluster_labels:
            it.setVisible(False)
        self.viewport().update()                     # drop the edges for the slide
        self._ta, self._tb, self._dur = a, b, float(dur)
        self._t0 = time.monotonic()
        self.timer.start(16)

    def _tween_step(self):
        t = (time.monotonic() - self._t0) / self._dur
        if t >= 1.0:
            self.pos = self._tb
            self._place_dots()
            self._place_trail()
            for it in self.trail_items + self.cluster_labels:
                it.setVisible(True)
            self.timer.stop()
            self._tweening = False
            self._rebuild_edge_lines()               # snap the plexus back onto the settled dots
            self.fit()
            return
        e = 1 - (1 - t) ** 3                         # ease-out cubic
        self.pos = self._ta + (self._tb - self._ta) * e
        self._place_dots()

    def _build_edges(self):
        """Compute the edge index array for the current CONNECT mode (drawn in drawBackground)."""
        self.edges = []
        mode, k, T, base = self.mode, self.density, self.tracks, self.base
        if mode == "none":
            self._edge_arr = None
            return
        pairs = set()
        if mode == "sonic":
            for i in range(len(T)):
                for j in self._knn[i][:k]:
                    pairs.add((min(i, j), max(i, j)))
        elif mode == "key":
            by_key = defaultdict(list)                 # bucket tracks by Camelot key ONCE
            for j, t in enumerate(T):
                if t.key:
                    by_key[t.key].append(j)
            for i in range(len(T)):
                if not T[i].key:
                    continue
                comp = library.camelot_neighbors(T[i].key)   # only the ~handful of compatible keys
                cand = np.fromiter((j for ck in comp for j in by_key.get(ck, []) if j != i),
                                   int)
                if not len(cand):
                    continue
                d = ((base[cand] - base[i]) ** 2).sum(1)     # vectorized spatial distance
                for j in cand[np.argsort(d)[:k]]:            # k nearest key-compatible tracks
                    j = int(j)
                    pairs.add((min(i, j), max(i, j)))
        elif mode == "tempo":
            order = [i for i in sorted(range(len(T)), key=lambda x: T[x].bpm or 1e9) if T[i].bpm]
            for a in range(len(order)):
                for b in range(a + 1, min(a + 1 + k, len(order))):
                    pairs.add((min(order[a], order[b]), max(order[a], order[b])))
        elif mode == "artist":
            groups = defaultdict(list)
            for i, t in enumerate(T):
                groups[_primary_artist(t.artist)].append(i)
            for art, mem in groups.items():
                if not art or len(mem) < 2:
                    continue
                for a in range(len(mem)):
                    for b in range(a + 1, min(a + 1 + k, len(mem))):
                        pairs.add((mem[a], mem[b]))
        self.edges = list(pairs)
        self._edge_arr = np.array(self.edges) if self.edges else None

    def reset_layout(self):
        """Snap dots back to their raw UMAP coordinates and re-frame (RESET button)."""
        self.timer.stop()
        self._tweening = False
        self.pos = self.base.copy() * S
        self._place_all()
        self.fit()

    # --- interaction ---
    def _node_at(self, pos):
        # scan all items under the cursor for the dot, so a cluster label / edge painted on top
        # can't swallow the click (labels are click-through but itemAt() still returns the topmost)
        for item in self.items(pos):
            if isinstance(item, QGraphicsEllipseItem):
                return item.data(0)
        return None

    def mousePressEvent(self, e):
        # middle-drag = jog/scrub the track; click a dot = play + open its modal;
        # click empty space = hand-drag to pan.
        if e.button() == Qt.MiddleButton:
            self._jog = e.position().x()
            self._jog_base = self.host.player.position()
            self.setCursor(Qt.SizeHorCursor)
            e.accept()
            return
        if e.button() == Qt.LeftButton:
            i = self._node_at(e.pos())
            if i is not None:
                self.host.select_track(self.tracks[i])   # host plays it + updates the bottom bar
                e.accept()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._jog is not None:
            dx = e.position().x() - self._jog
            dur = self.host.player.duration()
            self.host.player.setPosition(max(0, min(dur, int(self._jog_base + dx * JOG_MS_PER_PX))))
            e.accept()
            return
        i = self._node_at(e.pos())                 # hover a dot -> INSPECT (never disturbs playback)
        if i != self._hover:
            self._hover = i
            if i is not None:
                self.host.inspect(self.tracks[i])
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self._jog is not None:
            self._jog = None
            self.unsetCursor()
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e):
        f = 1.18 if e.angleDelta().y() > 0 else 1 / 1.18
        self.scale(f, f)
        e.accept()

    def _visible_rect(self) -> QRectF:
        """Scene-space bounding rect of the currently-visible dots (so subset views frame right)."""
        xs, ys = [], []
        for i, it in enumerate(self.dot_items):
            if it.isVisible():
                x, y = self._to_scene(self.pos[i])
                xs.append(x)
                ys.append(y)
        if not xs:
            return self.scene.itemsBoundingRect()
        pad = 40
        return QRectF(min(xs) - pad, min(ys) - pad,
                      max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad)

    def fit(self):
        # Pad the scene rect well beyond the content so there's always room to drag-pan,
        # then frame the visible dots. fitInView needs the Qt enum (an int silently hangs).
        r = self._visible_rect()
        self.setSceneRect(r.adjusted(-r.width(), -r.height(), r.width(), r.height()))
        self.fitInView(r, Qt.KeepAspectRatio)

    def set_subset(self, paths):
        """Show only dots whose path is in `paths` (None = show all); the plexus drops any edge
        touching a hidden node; refit. Drives both the crate lens and the bucket/search filters."""
        self._subset = set(paths) if paths is not None else None
        for i, it in enumerate(self.dot_items):
            it.setVisible(self._subset is None or self.tracks[i].path in self._subset)
        self._rebuild_edge_lines()
        self.fit()

    def nearest_unplayed(self, path, played):
        """Nearest track to `path` in UMAP space whose path isn't in `played` (and is within the
        current subset, if any). Forward-only MAP-walk helper for the host. Returns a Track/None."""
        if not len(self.base):
            return None
        i = next((k for k, t in enumerate(self.tracks) if t.path == path), None)
        if i is None:
            return None
        d = ((self.base - self.base[i]) ** 2).sum(1)
        for j in np.argsort(d):
            t = self.tracks[j]
            if j == i or t.path in played:
                continue
            if self._subset is not None and t.path not in self._subset:
                continue
            return t
        return None

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_fitted", False):
            self.fit()
            self._fitted = True


def _artist_color(name: str) -> QColor:
    """Stable-ish hue per artist name (just for visual separation on the scatter)."""
    h = (sum(ord(c) for c in name) * 47) % 360
    return QColor.fromHsvF(h / 360.0, 0.45, 1.0)


class ArtistMapView(QGraphicsView):
    """Artist-level UMAP scatter — a coarser 'who sits near whom' lens over the track map.

    Each dot = one artist, positioned by the mean of their tracks' CLAP vectors
    (`umap_artists.py` → `artist_umap.sqlite`) and SIZED by their track count. Hover = highlight
    + readout; click = filter the library list down to that artist (host switches to LIST).
    """

    def __init__(self, artists_xy, host):
        super().__init__()
        self.host = host
        # artists_xy: [(artist, x, y, n)]; x,y already in [0,1]
        self.artists = [a for a, x, y, n in artists_xy]
        self.ns = [int(n) for a, x, y, n in artists_xy]
        self.base = np.array([[x, y] for a, x, y, n in artists_xy], dtype=float)
        self._hover = None

        self.scene = QGraphicsScene(0, 0, W, H)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor("#05060a"))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)

        nmax = max(1, max(self.ns) if self.ns else 1)
        self._radii = []
        self.dot_items, self.label_items = [], []
        for i, a in enumerate(self.artists):
            r = 5.0 + 12.0 * float(np.sqrt(self.ns[i] / nmax))
            self._radii.append(r)
            c = _artist_color(a)
            it = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            it.setData(0, i)
            it.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 150)))
            pen = QPen(QColor(c.red(), c.green(), c.blue(), 235))
            pen.setWidthF(1.4)
            it.setPen(pen)
            it.setZValue(1)
            it.setToolTip(f"{a}  ·  {self.ns[i]} track{'s' if self.ns[i] != 1 else ''}")
            self.scene.addItem(it)
            self.dot_items.append(it)
            # names are ALWAYS visible (dim) so you can read the map without hovering; hover just
            # brightens + raises one. Labels never intercept clicks (see _node_at -> items()).
            lbl = QGraphicsTextItem(a)
            lbl.setDefaultTextColor(QColor(150, 160, 180))
            lbl.setAcceptHoverEvents(False)
            lbl.setAcceptedMouseButtons(Qt.NoButton)
            f = lbl.font()
            f.setPointSizeF(6.5)
            lbl.setFont(f)
            lbl.setOpacity(0.62)
            lbl.setZValue(2)
            self.scene.addItem(lbl)
            self.label_items.append(lbl)
        self._place()

    def _to_scene(self, p):
        return (M + p[0] * (W - 2 * M), M + p[1] * (H - 2 * M))

    def _place(self):
        for i, it in enumerate(self.dot_items):
            x, y = self._to_scene(self.base[i])
            it.setPos(x, y)
            self.label_items[i].setPos(x + self._radii[i] + 2, y - 9)

    def _node_at(self, pos):
        # scan ALL items under the cursor (topmost first) for the dot — so a label painted over a
        # dot can't swallow the click/hover (labels are click-through, but itemAt() still returns them)
        for item in self.items(pos):
            if isinstance(item, QGraphicsEllipseItem):
                return item.data(0)
        return None

    def _set_hover(self, i, on):
        it = self.dot_items[i]
        lbl = self.label_items[i]
        c = _artist_color(self.artists[i])
        pen = it.pen()
        if on:
            pen.setColor(QColor(255, 255, 255, 255))
            pen.setWidthF(2.6)
            it.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 220)))
            it.setZValue(5)
            lbl.setZValue(6)
            lbl.setOpacity(1.0)
            lbl.setDefaultTextColor(QColor(255, 255, 255))
        else:
            pen.setColor(QColor(c.red(), c.green(), c.blue(), 235))
            pen.setWidthF(1.4)
            it.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 150)))
            it.setZValue(1)
            lbl.setZValue(2)
            lbl.setOpacity(0.62)
            lbl.setDefaultTextColor(QColor(150, 160, 180))
        it.setPen(pen)

    def mouseMoveEvent(self, e):
        i = self._node_at(e.pos())
        if i != self._hover:
            if self._hover is not None:
                self._set_hover(self._hover, False)
            self._hover = i
            if i is not None:
                self._set_hover(i, True)
                self.host.inspect_artist(self.artists[i], self.ns[i])
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        if self._hover is not None:
            self._set_hover(self._hover, False)
            self._hover = None
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            i = self._node_at(e.pos())
            if i is not None:
                self.host.filter_to_artist(self.artists[i])
                e.accept()
                return
        super().mousePressEvent(e)

    def wheelEvent(self, e):
        f = 1.18 if e.angleDelta().y() > 0 else 1 / 1.18
        self.scale(f, f)
        e.accept()

    def fit(self):
        xs, ys = [], []
        for i in range(len(self.dot_items)):
            x, y = self._to_scene(self.base[i])
            xs.append(x)
            ys.append(y)
        if not xs:
            return
        pad = 60
        r = QRectF(min(xs) - pad, min(ys) - pad,
                   max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad)
        self.setSceneRect(r.adjusted(-r.width(), -r.height(), r.width(), r.height()))
        self.fitInView(r, Qt.KeepAspectRatio)

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_fitted", False):
            self.fit()
            self._fitted = True


class HistoryDialog(QDialog):
    """Shared listening-history log (used by both the main window and the map).

    `host.history` is a list of Tracks, newest first. Double-click an entry to replay it.
    """

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.setObjectName("card")
        self.setWindowTitle("LISTENING HISTORY")
        self.setModal(False)
        self.resize(470, 430)
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        hdr = QLabel("+-- LISTENING HISTORY " + "-" * 28 + "+")
        hdr.setObjectName("panelHeader")
        v.addWidget(hdr)
        self.list = QListWidget()
        self.list.itemActivated.connect(self._play)     # double-click / Enter
        v.addWidget(self.list, 1)
        hint = QLabel("double-click to replay")
        hint.setObjectName("readout")
        v.addWidget(hint)
        self.refresh()

    def refresh(self):
        self.list.clear()
        for t in self.host.history:
            bpm = f"{t.bpm:.0f}" if t.bpm else "—"
            it = QListWidgetItem(f"{t.artist} — {t.title}    [{t.key or '—'} · {bpm} BPM]")
            it.setData(Qt.UserRole, t)
            self.list.addItem(it)

    def _play(self, item):
        t = item.data(Qt.UserRole)
        if t:
            self.host.preview_track(t)


class MapView3D(QGraphicsView):
    """Software-projected 3D scatter of the coords3d UMAP — the galaxy you can orbit.

    Positions are projected from a centered Nx3 array through a camera rotation (azimuth around the
    vertical, elevation around the horizontal) on every update; camera-Z depth drives each dot's
    size + brightness and its paint order, so the cloud reads as real 3D with NO OpenGL dependency
    (a pure PySide6 + numpy widget). Drag = orbit, wheel = zoom, click a dot = play it. CONNECT
    becomes a 3D shape: Sonic/None = the UMAP cloud; Tempo = a vertical DNA helix (slow bottom ->
    fast top); Key = a flat Camelot wheel you can tilt.
    """

    def __init__(self, tracks_xyz, host):
        super().__init__()
        self.host = host
        self.tracks = [t for t, x, y, z in tracks_xyz]
        b = np.array([[x, y, z] for t, x, y, z in tracks_xyz], dtype=float)
        self.base = (b - b.mean(0)) if len(b) else b.reshape(0, 3)   # centered ~[-0.5, 0.5]
        self.mode = "sonic"
        self.color_mode = "cluster"
        self.az, self.el, self.zoom = 30.0, 18.0, 1.0
        self.playing_path = None
        self._idx_of = {t.path: i for i, t in enumerate(self.tracks)}   # path -> dot index
        self.trail_idx = []        # listening trail as dot indices, oldest -> newest
        self._subset = None
        self._p3 = self.base.copy()
        self._scr = np.zeros((len(self.tracks), 2))
        self._press = None
        self._moved = False
        self._jog = None
        self._jog_base = 0
        self._hover = None
        self._edges = None        # list of (i, j) index pairs for the active CONNECT plexus
        self._k = DENSITY["Sparse"]   # plexus edges per node — driven by the DENSITY selector
        try:
            self.clusters = library.load_clusters()
        except Exception:
            self.clusters = {}
        self.scene = QGraphicsScene(0, 0, W, H)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor("#05060a"))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.dot_items = []
        for i, t in enumerate(self.tracks):
            it = QGraphicsEllipseItem(-R, -R, 2 * R, 2 * R)
            it.setData(0, i)
            it.setToolTip(rich_info(t))
            self.scene.addItem(it)
            self.dot_items.append(it)
        self._rebuild_edges()
        self._update()

    def _rot(self):
        a, e = math.radians(self.az), math.radians(self.el)
        ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
        return rx @ ry

    def _color(self, t) -> QColor:
        m = self.color_mode
        if m == "tempo":
            return _tempo_color(t.bpm)
        if m == "energy":
            return _energy_color(t.energy)
        if m == "danceability":
            return _dance_color(getattr(t, "danceability", None))
        if m == "artist":
            return _artist_color(_primary_artist(t.artist))
        if m == "key":
            return key_color(t.key)
        return _cluster_color(self.clusters.get(t.path))

    def _update(self):
        if not len(self._p3):
            return
        pr = self._p3 @ self._rot().T
        z = pr[:, 2]
        zn = (z - z.min()) / (np.ptp(z) + 1e-9)
        s = min(W, H) * 0.70 * self.zoom
        cx, cy = W / 2.0, H / 2.0
        self._scr = np.column_stack([cx + pr[:, 0] * s, cy - pr[:, 1] * s])
        for i, it in enumerate(self.dot_items):
            if self._subset is not None and self.tracks[i].path not in self._subset:
                it.setVisible(False)
                continue
            it.setVisible(True)
            depth = float(zn[i])
            r = R * (0.55 + 0.9 * depth) + (self.tracks[i].rating or 0) * 0.5
            it.setRect(-r, -r, 2 * r, 2 * r)
            it.setPos(float(self._scr[i, 0]), float(self._scr[i, 1]))
            c = self._color(self.tracks[i])
            if self.tracks[i].path == self.playing_path:   # now playing -> bright fill + white halo
                it.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 255)))
                pen = QPen(QColor(255, 255, 255, 255)); pen.setWidthF(3.0)
                it.setPen(pen); it.setZValue(1e6)
            else:
                c = QColor(c); c.setAlpha(int(70 + 170 * depth))
                it.setBrush(QColor(0, 0, 0, 0))
                pen = QPen(c); pen.setWidthF(1.6)
                it.setPen(pen); it.setZValue(float(z[i]))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)   # scale the scene into the viewport
        self.viewport().update()                                     # repaint the projected plexus (drawBackground)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def drawBackground(self, painter, rect):
        # Project the CONNECT plexus every frame; the dots (scene items) paint on top. self._scr is in
        # scene coords — the same space as drawBackground's painter — so no extra transform is needed.
        super().drawBackground(painter, rect)
        if not len(self._scr):
            return
        scr, sub = self._scr, self._subset
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self._edges and self.mode != "none":
            base = EDGE_COLOR.get(self.mode, QColor(180, 210, 255, 55))   # dim the plexus in 3D — overlapping
            col = QColor(base.red(), base.green(), base.blue(),           # projected lines compound otherwise
                         max(16, int(base.alpha() * 0.45)))
            pen = QPen(col)
            pen.setWidthF(0.8)
            pen.setCosmetic(True)
            painter.setPen(pen)
            lines = []
            for i, j in self._edges:
                if sub is not None and (self.tracks[i].path not in sub or self.tracks[j].path not in sub):
                    continue
                lines.append(QLineF(scr[i, 0], scr[i, 1], scr[j, 0], scr[j, 1]))
            if lines:
                painter.drawLines(lines)
        self._draw_trail(painter)

    def _draw_trail(self, painter):
        """Comet line threading the played dots in order, brightening + thickening toward the most
        recent (projected through the same camera as the dots)."""
        idx, sub = self.trail_idx, self._subset
        n = len(idx)
        if n < 2:
            return
        scr = self._scr
        for s in range(n - 1):
            i, j = idx[s], idx[s + 1]
            if sub is not None and (self.tracks[i].path not in sub or self.tracks[j].path not in sub):
                continue
            frac = (s + 1) / (n - 1)                       # 0..1, brighter/thicker toward newest
            pen = QPen(QColor(180, 210, 255, int(35 + 165 * frac)))
            pen.setWidthF(1.0 + 1.8 * frac)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(QLineF(scr[i, 0], scr[i, 1], scr[j, 0], scr[j, 1]))

    def _rebuild_edges(self):
        """Index pairs for the current CONNECT mode, mirroring the 2D MapView plexus but drawn as
        projected 3D lines. Sonic = vector KNN re-ranked by mixability; key/tempo/artist = the same
        relations the flat map uses. Recomputed on mode change (cheap); the draw pass filters subset."""
        k, T = max(1, self._k // 2), self.tracks   # 3D projects edges on top of each other, so half
        mode = self.mode                            # the 2D density reads as plenty (Sparse->1, Medium->2)
        pairs = set()
        if mode == "none" or not T:
            self._edges = None
            return
        if mode == "sonic":
            knn = self._sonic_knn(k)
            if knn:
                pairs = knn
        elif mode == "key":
            by_key = defaultdict(list)
            for j, t in enumerate(T):
                if t.key:
                    by_key[t.key].append(j)
            for i in range(len(T)):
                if not T[i].key:
                    continue
                comp = library.camelot_neighbors(T[i].key)
                cand = np.fromiter((j for ck in comp for j in by_key.get(ck, []) if j != i), int)
                if not len(cand):
                    continue
                d = ((self._p3[cand] - self._p3[i]) ** 2).sum(1)   # nearest key-compatible on the wheel
                for j in cand[np.argsort(d)[:k]]:
                    j = int(j)
                    pairs.add((min(i, j), max(i, j)))
        elif mode == "tempo":
            order = [i for i in sorted(range(len(T)), key=lambda x: T[x].bpm or 1e9) if T[i].bpm]
            for a in range(len(order)):
                for b in range(a + 1, min(a + 1 + k, len(order))):   # chain consecutive BPM = the strand
                    pairs.add((min(order[a], order[b]), max(order[a], order[b])))
        elif mode == "artist":
            groups = defaultdict(list)
            for i, t in enumerate(T):
                groups[_primary_artist(t.artist)].append(i)
            for art, mem in groups.items():
                if not art or len(mem) < 2:
                    continue
                for a in range(len(mem)):
                    for b in range(a + 1, min(a + 1 + k, len(mem))):
                        pairs.add((mem[a], mem[b]))
        self._edges = list(pairs) if pairs else None

    def _sonic_knn(self, k):
        """Vector-space KNN (centered 512-d) re-ranked by DJ mixability — the sonic plexus, as in 2D."""
        try:
            vecs = library.load_vectors()
        except Exception:
            vecs = {}
        paths = [t.path for t in self.tracks]
        have = [i for i, p in enumerate(paths) if p in vecs]
        if len(have) < max(3, 0.5 * len(paths)):
            return None
        M = np.stack([vecs[paths[i]] for i in have])
        sims = M @ M.T
        row_of = {i: a for a, i in enumerate(have)}
        pairs = set()
        for a, i in enumerate(have):
            order = [have[b] for b in np.argsort(-sims[a]) if have[b] != i]
            cand = order[:k * KNN_OVERSAMPLE]
            ta = self.tracks[i]
            cand.sort(key=lambda j: -(float(sims[a][row_of[j]])
                                      - KNN_PENALTY_W * library.compat_penalty(ta, self.tracks[j])))
            for j in cand[:k]:
                pairs.add((min(i, j), max(i, j)))
        return pairs

    def _shape(self, mode):
        n = len(self.tracks)
        if mode in ("none", "sonic", "artist") or n == 0:
            return self.base.copy()
        if mode == "tempo":                               # a loose, irregular DNA strand (slow->fast up)
            P = self.base.copy()
            rng = np.random.default_rng(42)
            order = [i for i in sorted(range(n), key=lambda x: self.tracks[x].bpm or 1e9)
                     if self.tracks[i].bpm]
            m = len(order)
            turns, rad, tlt = 7.0, 0.40, math.radians(14)
            ct, st = math.cos(tlt), math.sin(tlt)
            for rank, i in enumerate(order):
                f = rank / max(1, m - 1)
                th = f * turns * 2 * math.pi + rng.uniform(-0.18, 0.18)        # angular wobble
                rr = rad + rng.uniform(-0.06, 0.06) + 0.05 * math.sin(f * 11)  # breathing radius
                yy = (f - 0.5) * 1.9 + rng.uniform(-0.015, 0.015)             # taller column + jitter
                x, y, z = rr * math.cos(th), yy, rr * math.sin(th)
                P[i] = [x * ct - y * st, x * st + y * ct, z]                  # lean the column slightly
            for i, t in enumerate(self.tracks):
                if not t.bpm:
                    P[i] = [rng.uniform(-0.04, 0.04), -1.05, rng.uniform(-0.04, 0.04)]
            return P
        if mode == "key":                                 # Camelot wheel with real thickness (3D puffs)
            P = self.base.copy()
            rng = np.random.default_rng(7)
            r_in, r_out = 0.34, 0.54
            by_key = defaultdict(list)
            for i, t in enumerate(self.tracks):
                p = library._parse_camelot(t.key) if t.key else None
                if not p:
                    P[i] = [rng.uniform(-0.05, 0.05), 0.72, rng.uniform(-0.05, 0.05)]
                    continue
                by_key[p].append(i)
            for (num, letter), members in by_key.items():
                ang = (num - 1) / 12.0 * 2 * math.pi - math.pi / 2
                rad = r_in if letter == "A" else r_out               # inner ring = minor (A), outer = major (B)
                for s, i in enumerate(members):
                    ja = rng.uniform(-0.16, 0.16)                    # spread within the key's clump...
                    jr = rng.uniform(-0.045, 0.045)
                    pz = (s / max(1, len(members) - 1) - 0.5) * 0.30 + rng.uniform(-0.02, 0.02)  # ...+ z column
                    P[i] = [(rad + jr) * math.cos(ang + ja), (rad + jr) * math.sin(ang + ja), pz]
            return P
        return self.base.copy()

    def set_mode(self, mode, k=None):
        self.mode = mode
        if k:
            self._k = k                                   # DENSITY selector applies in 3D too
        self._p3 = self._shape(mode)
        self._rebuild_edges()
        self._update()

    def nearest_unplayed(self, path, played):
        """Nearest track to `path` in the 3D galaxy whose path isn't in `played` (and is within the
        current subset). Forward-only MAP-walk helper, mirroring 2D MapView but in coords3d space so
        the walk follows what you actually SEE in the orbit. Returns a Track / None."""
        if not len(self.base):
            return None
        i = next((k for k, t in enumerate(self.tracks) if t.path == path), None)
        if i is None:
            return None
        d = ((self.base - self.base[i]) ** 2).sum(1)
        for j in np.argsort(d):
            t = self.tracks[j]
            if j == i or t.path in played:
                continue
            if self._subset is not None and t.path not in self._subset:
                continue
            return t
        return None

    def set_color_mode(self, mode):
        self.color_mode = mode
        self._update()

    def set_playing(self, path):
        self.playing_path = path
        self._update()

    def set_trail(self, paths):
        """Listening trail as a projected comet through the played dots (oldest -> newest). `paths`
        comes from the host history; we keep only known dots and redraw via drawBackground."""
        self.trail_idx = [self._idx_of[p] for p in paths if p in self._idx_of]
        self.viewport().update()

    def set_subset(self, paths):
        # `paths is not None` (not truthiness): an EMPTY set = a zero-result filter that should hide
        # every dot, not fall through to "show all" (which `if paths` would do). Mirrors 2D MapView.
        self._subset = set(paths) if paths is not None else None
        self._update()

    def fit(self):
        self.az, self.el, self.zoom = 30.0, 18.0, 1.0
        self._update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:                  # middle-drag = jog/scrub (leave orbit alone)
            self._jog = e.position().x()
            self._jog_base = self.host.player.position()
            self.setCursor(Qt.SizeHorCursor)
            e.accept()
            return
        if e.button() == Qt.LeftButton:                    # left-drag = orbit
            self._press = e.position()
            self._moved = False
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._jog is not None:                          # scrub the playing track L/R
            dx = e.position().x() - self._jog
            dur = self.host.player.duration()
            self.host.player.setPosition(max(0, min(dur, int(self._jog_base + dx * JOG_MS_PER_PX))))
            e.accept()
            return
        if self._press is not None:                        # orbit
            d = e.position() - self._press
            if abs(d.x()) + abs(d.y()) > 3:
                self._moved = True
            self.az += d.x() * 0.4
            self.el = max(-85.0, min(85.0, self.el + d.y() * 0.4))
            self._press = e.position()
            self._update()
            return
        i = self._pick(e.position())                       # hover -> INSPECT readout (read-only:
        if i != self._hover:                               # never changes the selection that ＋ADD /
            self._hover = i                                # COMPATIBLE act on — that's the clicked dot)
            if i is not None:
                self.host.inspect(self.tracks[i], select=False)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self._jog is not None:
            self._jog = None
            self.unsetCursor()
            e.accept()
            return
        if e.button() == Qt.LeftButton and self._press is not None and not self._moved:
            i = self._pick(e.position())                   # a click (no orbit) -> play that dot
            if i is not None:
                self.host.select_track(self.tracks[i])
        self._press = None

    def _pick(self, pos):
        """Nearest visible dot to a viewport-space point, within 18 px. Compares in VIEWPORT space
        (project the scene-space dot centres through the view transform) so the pick lines up with the
        cursor regardless of the fitInView scale — the old code tested viewport px against scene coords,
        which drifted the hit up/right of where you actually clicked."""
        if not len(self._scr):
            return None
        t = self.viewportTransform()
        vx = t.m11() * self._scr[:, 0] + t.dx()
        vy = t.m22() * self._scr[:, 1] + t.dy()
        d2 = (vx - pos.x()) ** 2 + (vy - pos.y()) ** 2
        if self._subset is not None:
            mask = np.fromiter((self.tracks[i].path in self._subset for i in range(len(self.tracks))),
                               bool, len(self.tracks))
            d2 = np.where(mask, d2, np.inf)
        j = int(np.argmin(d2))
        return j if d2[j] <= 18 * 18 else None

    def wheelEvent(self, e):
        self.zoom *= 1.0015 ** e.angleDelta().y()
        self.zoom = max(0.25, min(6.0, self.zoom))
        self._update()


class ArtistMapView3D(QGraphicsView):
    """Orbitable 3D artist galaxy — the 3D twin of ArtistMapView, over the `artists3d` UMAP.

    Same software projection as MapView3D (project an Nx3 array through an az/el camera each frame;
    camera-Z drives size + brightness + paint order). Dot size = track count, hue per artist, names
    always visible (dim, brighten on hover). Drag = orbit, wheel = zoom, click an artist = filter the
    library list to it (host drops to LIST). CONNECT / COLOR BY / DENSITY don't apply to this grain.
    """

    def __init__(self, artists_xyz, host):
        super().__init__()
        self.host = host
        self.artists = [a for a, x, y, z, n in artists_xyz]
        self.ns = [int(n) for a, x, y, z, n in artists_xyz]
        b = np.array([[x, y, z] for a, x, y, z, n in artists_xyz], dtype=float)
        self.base = (b - b.mean(0)) if len(b) else b.reshape(0, 3)   # centered ~[-0.5, 0.5]
        self._p3 = self.base.copy()
        self.az, self.el, self.zoom = 30.0, 18.0, 1.0
        self._press, self._moved, self._hover = None, False, None
        self._scr = np.zeros((len(self.artists), 2))
        nmax = max(1, max(self.ns) if self.ns else 1)
        self._radii = [5.0 + 12.0 * float(np.sqrt(n / nmax)) for n in self.ns]
        self.scene = QGraphicsScene(0, 0, W, H)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor("#05060a"))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.dot_items, self.label_items = [], []
        for i, a in enumerate(self.artists):
            r = self._radii[i]
            it = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            it.setData(0, i)
            it.setToolTip(f"{a}  ·  {self.ns[i]} track{'s' if self.ns[i] != 1 else ''}")
            self.scene.addItem(it)
            self.dot_items.append(it)
            lbl = QGraphicsTextItem(a)               # names always visible (dim), click-through
            lbl.setAcceptHoverEvents(False)
            lbl.setAcceptedMouseButtons(Qt.NoButton)
            f = lbl.font()
            f.setPointSizeF(6.5)
            lbl.setFont(f)
            lbl.setZValue(2)
            self.scene.addItem(lbl)
            self.label_items.append(lbl)
        self._update()

    def _rot(self):
        a, e = math.radians(self.az), math.radians(self.el)
        ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
        return rx @ ry

    def _update(self):
        if not len(self._p3):
            return
        pr = self._p3 @ self._rot().T
        z = pr[:, 2]
        zn = (z - z.min()) / (np.ptp(z) + 1e-9)
        s = min(W, H) * 0.66 * self.zoom
        cx, cy = W / 2.0, H / 2.0
        self._scr = np.column_stack([cx + pr[:, 0] * s, cy - pr[:, 1] * s])
        for i, it in enumerate(self.dot_items):
            depth = float(zn[i])
            r = self._radii[i] * (0.5 + 0.85 * depth)
            it.setRect(-r, -r, 2 * r, 2 * r)
            it.setPos(float(self._scr[i, 0]), float(self._scr[i, 1]))
            c = _artist_color(self.artists[i])
            hov = (i == self._hover)
            fa = 235 if hov else int(70 + 150 * depth)
            it.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), fa)))
            pen = QPen(QColor(255, 255, 255, 255) if hov
                       else QColor(c.red(), c.green(), c.blue(), min(255, fa + 60)))
            pen.setWidthF(2.6 if hov else 1.4)
            it.setPen(pen)
            it.setZValue(float(z[i]) + (1e6 if hov else 0.0))
            lbl = self.label_items[i]
            lbl.setPos(float(self._scr[i, 0]) + r + 2, float(self._scr[i, 1]) - 9)
            lbl.setDefaultTextColor(QColor(255, 255, 255) if hov else QColor(150, 160, 180))
            lbl.setOpacity(1.0 if hov else (0.3 + 0.5 * depth))
            lbl.setZValue(float(z[i]) + (1e6 if hov else 0.5))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    # CONNECT / COLOR BY / DENSITY / track-playback don't apply to the artist grain — accept + ignore
    # so the host can route to any active map uniformly.
    def set_color_mode(self, mode): pass
    def set_mode(self, mode, k=None): pass
    def set_subset(self, paths): pass
    def set_playing(self, path): pass
    def set_trail(self, paths): pass

    def fit(self):
        self.az, self.el, self.zoom = 30.0, 18.0, 1.0
        self._update()

    def _pick(self, pos):
        if not len(self._scr):
            return None
        t = self.viewportTransform()
        vx = t.m11() * self._scr[:, 0] + t.dx()
        vy = t.m22() * self._scr[:, 1] + t.dy()
        d2 = (vx - pos.x()) ** 2 + (vy - pos.y()) ** 2
        j = int(np.argmin(d2))
        return j if d2[j] <= 22 * 22 else None

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.position()
            self._moved = False
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._press is not None:                        # orbit
            d = e.position() - self._press
            if abs(d.x()) + abs(d.y()) > 3:
                self._moved = True
            self.az += d.x() * 0.4
            self.el = max(-85.0, min(85.0, self.el + d.y() * 0.4))
            self._press = e.position()
            self._update()
            return
        i = self._pick(e.position())                       # hover -> highlight + readout
        if i != self._hover:
            self._hover = i
            if i is not None:
                self.host.inspect_artist(self.artists[i], self.ns[i])
            self._update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._press is not None and not self._moved:
            i = self._pick(e.position())                   # a click (no orbit) -> filter to that artist
            if i is not None:
                self.host.filter_to_artist(self.artists[i])
        self._press = None

    def wheelEvent(self, e):
        self.zoom *= 1.0015 ** e.angleDelta().y()
        self.zoom = max(0.25, min(6.0, self.zoom))
        self._update()
