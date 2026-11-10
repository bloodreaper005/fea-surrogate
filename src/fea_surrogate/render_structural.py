"""Ansys-Mechanical-style pictures of a static structural analysis.

Three pictures a person needs, in the order they need them:

* `render_face_ids` -- before the interview: every CAD face numbered, from two
  opposite iso views, so "which face is held?" has an answer you can point at.
* `render_setup` -- before solving: the environment view, supports and loads
  painted on the part and lettered A, B, C... exactly as the confirmation box
  letters them. A load on the wrong face is obvious here and invisible in JSON.
* `render_result` -- after solving: Equivalent (von-Mises) Stress or Total
  Deformation on the deformed shape, drawn the way an Ansys user expects to read
  it -- 9-band rainbow legend with Max/Min, title block, deformation scale,
  undeformed wireframe, probes -- because that is the picture the result will be
  compared against.

A solved picture is allowed to look solved: it carries `structural_header`
("SOLVED -- linear elastic ...") and the rainbow. It must never carry the
screening words, and the screening renderer (`render.py`, cividis, ESTIMATE
watermark) must never borrow this palette; `ANSYS_LEGEND_COLORS` is a separate
constant on purpose.

Why the drawing is a hand-rolled orthographic projection onto a 2D canvas rather
than `mplot3d`: mplot3d clips the part to a square axes box (long parts lose
their ends -- the bug in `render_solved`), cannot depth-test lines against
surfaces (hidden edges show through), and does not expose a pixel scale, so a
scale ruler would be a guess. Projecting ourselves gives exact back-face culling
(the boundary winding is outward), a small numpy z-buffer that hides occluded
edges and decides which face labels are visible, a framing that always contains
the whole part, and an exact millimetre ruler. Everything is drawn in figure
pixels on one full-figure axes with clipping off.

Heavy imports (matplotlib) stay inside functions: importing this module costs
numpy only.
"""

from __future__ import annotations

import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


#: Ansys Mechanical's default 9-band contour legend, Min (blue) -> Max (red).
#: Deliberately its own constant: `render.FIELD_COLORMAP` is the screening
#: palette and must stay "not a rainbow", so the two can never be confused.
ANSYS_LEGEND_COLORS: tuple[str, ...] = (
    "#0000ff", "#00b2ff", "#00ffff", "#00ffb2", "#00ff00",
    "#b2ff00", "#ffff00", "#ffb200", "#ff0000",
)

#: (elevation, azimuth) in degrees, matplotlib's convention: the camera sits at
#: (cos e cos a, cos e sin a, sin e) from the model. (24, -58) is a classic
#: Z-up isometric-ish view from +X, -Y, +Z.
DEFAULT_VIEW: tuple[float, float] = (24.0, -58.0)

QUANTITIES: tuple[str, ...] = ("equivalent_stress", "total_deformation")

#: Ansys environment colours: supports blue-violet, loads red-family, gravity gold.
SUPPORT_COLOR = "#6a5acd"
LOAD_COLORS: tuple[str, ...] = ("#e0201b", "#ff7f0e", "#c2185b", "#8b0000", "#ff4f8b")
GRAVITY_COLOR = "#d4a017"
PART_GREY = "#c4c8ce"

#: Face-ID colours: mid-light and mutually distinct, none of them the result
#: rainbow, so a face picture is never mistaken for a contour plot.
FACE_PALETTE: tuple[str, ...] = (
    "#8dd3c7", "#ffd966", "#bebada", "#fb8072", "#80b1d3", "#fdb462",
    "#b3de69", "#fccde5", "#bc80bd", "#ccebc5", "#d9d9d9", "#e5c494",
)

LEGEND_LEVELS = 9


class StructuralRenderError(ValueError):
    """The picture cannot be drawn from what was passed; the message says why."""


@dataclass(frozen=True)
class _Result:
    key: str
    title: str
    type_line: str
    unit: str


_RESULTS: Mapping[str, _Result] = {
    "equivalent_stress": _Result(
        "equivalent_stress", "Equivalent Stress", "Equivalent (von-Mises) Stress", "MPa"
    ),
    "total_deformation": _Result(
        "total_deformation", "Total Deformation", "Total Deformation", "mm"
    ),
}


def structural_header(order: int) -> str:
    """The solved stamp burned into every result picture.

    Names the element so a reader can tell a 10-node (SOLID187-like) result from
    a 4-node comparison run, which is far too stiff in bending.
    """
    if order == 2:
        return "SOLVED — linear elastic, 10-node tetrahedra"
    if order == 1:
        # The caveat travels with the picture: a 4-node result screenshotted
        # out of the run reads bending stress about half of the truth.
        return "SOLVED — linear elastic, 4-node tetrahedra (stiff in bending: comparison only)"
    raise StructuralRenderError(f"element order must be 1 or 2, not {order!r}")


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------


def _num(value: float, sig: int = 4) -> str:
    """200000, 9.06, 0.0045 -- plain digits where they read well, else 1.2e-5."""
    value = float(value)
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e6:
        exponent = math.floor(math.log10(magnitude))
        decimals = max(0, sig - 1 - exponent)
        text = f"{value:.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text
    mantissa, exponent = f"{value:.{sig - 1}e}".split("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def _legend_num(value: float) -> str:
    """One label, 5 significant figures (see `_legend_labels` for a legend)."""
    return _num(value, 5)


def _legend_labels(values: Sequence[float]) -> list[str]:
    """Legend labels in ONE notation, chosen from the largest value: scientific
    for every label when it is below 0.01 or 1e5 and up, else plain digits for
    every label -- never 0.0011 above 9.1628e-4 in the same legend. 5
    significant figures each, as Ansys prints them."""
    values = [float(v) for v in values]
    big = max((abs(v) for v in values if math.isfinite(v)), default=0.0)
    if not big > 0:
        return [_num(v, 5) for v in values]
    scientific = big < 0.01 or big >= 1e5
    labels = []
    for v in values:
        if v == 0 or abs(v) < 1e-9 * big:
            labels.append("0")
        elif scientific:
            mantissa, exponent = f"{v:.4e}".split("e")
            mantissa = mantissa.rstrip("0").rstrip(".") if "." in mantissa else mantissa
            labels.append(f"{mantissa}e{int(exponent)}")
        else:
            decimals = max(0, 4 - math.floor(math.log10(abs(v))))
            text = f"{v:.{decimals}f}"
            labels.append(text.rstrip("0").rstrip(".") if "." in text else text)
    return labels


def _faces_phrase(tags: Sequence[int]) -> str:
    tags = list(tags)
    return ("face " if len(tags) == 1 else "faces ") + ", ".join(map(str, tags))


def _direction_name(vector: Sequence[float]) -> str:
    """'−Z' for an axis, else rounded components: what a person would type."""
    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    if not norm > 0:
        return "(0, 0, 0)"
    v = v / norm
    axis = int(np.argmax(np.abs(v)))
    if abs(abs(v[axis]) - 1.0) < 1e-6:
        return ("+" if v[axis] > 0 else "−") + "XYZ"[axis]
    return "(" + ", ".join(_num(c, 3) for c in v) + ")"


# --------------------------------------------------------------------------
# Geometry: camera, surface, z-buffer, edges
# --------------------------------------------------------------------------


def _basis(elev: float, azim: float) -> np.ndarray:
    """Rows: screen right, screen up, towards the viewer. Right-handed, so an
    outward triangle that faces the camera is counter-clockwise on screen."""
    e, a = math.radians(float(elev)), math.radians(float(azim))
    eye = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    right = np.array([-math.sin(a), math.cos(a), 0.0])
    up = np.cross(eye, right)
    return np.vstack([right, up, eye])


def _project(points: np.ndarray, view: tuple[float, float]) -> np.ndarray:
    """(n,3) model mm -> (n,3) view mm: x right, y up, z towards the viewer."""
    return np.asarray(points, dtype=float) @ _basis(*view).T


def _check_view(view: Any) -> tuple[float, float]:
    try:
        elev, azim = (float(v) for v in view)
    except (TypeError, ValueError):
        raise StructuralRenderError(
            f"view must be (elevation, azimuth) in degrees, e.g. {DEFAULT_VIEW}, not {view!r}"
        ) from None
    if not (math.isfinite(elev) and math.isfinite(azim)):
        raise StructuralRenderError(f"view angles must be finite, not {view!r}")
    return elev, azim


def _surface(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    """Linear surface triangles and the boundary row each came from.

    A tri6 is split into four with the gmsh ordering (3: 0-1, 4: 1-2, 5: 2-0):
    the midside nodes carry their own displacement and field values, so the
    deformed shape bends and the bands follow the quadratic field instead of
    being flattened across each element.
    """
    boundary = np.asarray(mesh.boundary, dtype=np.int64)
    if boundary.ndim != 2 or boundary.shape[1] not in (3, 6) or boundary.shape[0] == 0:
        raise StructuralRenderError(
            "mesh.boundary must be (F,3) or (F,6) surface triangles; this mesh has "
            f"shape {boundary.shape}. Mesh the part with structural_mesh.mesh_for_analysis"
        )
    rows = np.arange(boundary.shape[0])
    if boundary.shape[1] == 3:
        return boundary, rows
    b = boundary
    tris = np.concatenate([b[:, [0, 3, 5]], b[:, [3, 1, 4]], b[:, [5, 4, 2]], b[:, [3, 4, 5]]])
    return tris, np.tile(rows, 4)


def _area2(V: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Twice the signed screen area; > 0 means the outward side faces the camera."""
    a, b, c = V[tris[:, 0], :2], V[tris[:, 1], :2], V[tris[:, 2], :2]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _normals(V: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def _smooth_normals(V: np.ndarray, tris: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """Per-triangle shading normals averaged over each CAD face's vertices.

    Flat per-facet shading shows the tessellation of a bore or fillet as
    stripes. Averaging vertex normals only within one CAD face smooths curved
    faces while keeping the crease between two faces sharp.
    """
    raw = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    if len(tris) == 0:
        return raw
    face = np.repeat(np.asarray(owner, dtype=np.int64), 3)
    face = face - face.min()
    key = tris.reshape(-1).astype(np.int64) * (int(face.max()) + 1) + face
    _, inverse = np.unique(key, return_inverse=True)
    acc = np.zeros((int(inverse.max()) + 1, 3))
    np.add.at(acc, inverse, np.repeat(raw, 3, axis=0))
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-300)
    per_tri = acc[inverse].reshape(-1, 3, 3).sum(axis=1)
    return per_tri / np.maximum(np.linalg.norm(per_tri, axis=1, keepdims=True), 1e-300)


#: Light from upper left, slightly in front, fixed to the camera like a headlamp.
_LIGHT = np.array([-0.35, 0.55, 1.0]) / np.linalg.norm([-0.35, 0.55, 1.0])


def _shade(rgba: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Mild Lambert (0.70 + 0.30|n.l|): enough to read the shape, not enough to
    move a band's colour into its neighbour's."""
    out = np.array(rgba, dtype=float, copy=True)
    out[:, :3] *= (0.70 + 0.30 * np.abs(normals @ _LIGHT))[:, None]
    return out


class _DepthMap:
    """A small z-buffer of the front-facing surface, in view millimetres.

    matplotlib paints polygons in depth order but cannot hide a line behind a
    surface or say whether a point is visible. This answers both: hidden CAD
    edges are dropped, face labels are only drawn where the face can be seen,
    and the camera chooser can count visible area.
    """

    def __init__(self, V: np.ndarray, tris: np.ndarray, resolution: int = 700,
                 count: bool = False) -> None:
        self.tris = tris
        used = V[tris][..., :2].reshape(-1, 2) if len(tris) else V[:, :2]
        lo, hi = used.min(axis=0), used.max(axis=0)
        span = max(float((hi - lo).max()), 1e-9)
        self.pix = span / float(resolution)
        self.origin = lo - 2.0 * self.pix
        width = int(math.ceil((hi[0] - lo[0]) / self.pix)) + 5
        height = int(math.ceil((hi[1] - lo[1]) / self.pix)) + 5
        self.shape = (height, width)
        z = np.full(height * width, -np.inf)
        self.tri_pixels = np.zeros(len(tris), dtype=np.int64)
        #: Index into `tris` of the triangle seen at each pixel, -1 for none
        #: (filled when `count`): which face a pixel shows, for label placement.
        self.tri_at = np.full(height * width, -1, dtype=np.int64)
        if len(tris):
            p = (V[tris][..., :2] - self.origin) / self.pix
            depth = V[tris][..., 2]
            self._raster(p, depth, z, width, None)
            if count:
                self._raster(p, depth, z, width, self.tri_pixels, self.tri_at)
        self.z = z.reshape(height, width)
        self.tri_at = self.tri_at.reshape(height, width)
        padded = np.pad(self.z, 1, constant_values=-np.inf)
        loose = self.z.copy()
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                loose = np.minimum(loose, padded[dy:dy + height, dx:dx + width])
        #: Farthest surface in each 3x3 neighbourhood: a point on a crease or a
        #: silhouette is visible if it is not behind this.
        self.z_loose = loose

    @staticmethod
    def _raster(p: np.ndarray, depth: np.ndarray, z: np.ndarray, width: int,
                counter: np.ndarray | None, owner: np.ndarray | None = None) -> None:
        x0 = np.ceil(p[..., 0].min(axis=1) - 0.5).astype(np.int64)
        x1 = np.floor(p[..., 0].max(axis=1) - 0.5).astype(np.int64)
        y0 = np.ceil(p[..., 1].min(axis=1) - 0.5).astype(np.int64)
        y1 = np.floor(p[..., 1].max(axis=1) - 0.5).astype(np.int64)
        nx = np.maximum(x1 - x0 + 1, 0)
        ny = np.maximum(y1 - y0 + 1, 0)
        counts = nx * ny
        a, b, c = p[:, 0], p[:, 1], p[:, 2]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        area = np.where(np.abs(area) > 1e-12, area, np.nan)
        # Chunk so a few huge triangles on a coarse mesh cannot exhaust memory.
        limit = 2_000_000
        cumulative = np.cumsum(counts)
        start = 0
        while start < len(counts):
            base = cumulative[start - 1] if start else 0
            stop = int(np.searchsorted(cumulative, base + limit, side="right"))
            stop = max(stop, start + 1)
            sl = slice(start, stop)
            c_sl = counts[sl]
            total = int(c_sl.sum())
            start = stop
            if total == 0:
                continue
            tri = np.repeat(np.arange(sl.start, sl.stop), c_sl)
            local = np.arange(total) - np.repeat(np.cumsum(c_sl) - c_sl, c_sl)
            px = x0[tri] + local % nx[tri]
            py = y0[tri] + local // nx[tri]
            cx, cy = px + 0.5, py + 0.5
            ax_, ay = a[tri, 0], a[tri, 1]
            bx, by = b[tri, 0], b[tri, 1]
            qx, qy = c[tri, 0], c[tri, 1]
            ar = area[tri]
            w0 = ((bx - cx) * (qy - cy) - (by - cy) * (qx - cx)) / ar
            w1 = ((qx - cx) * (ay - cy) - (qy - cy) * (ax_ - cx)) / ar
            w2 = 1.0 - w0 - w1
            eps = -1e-6
            inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
            d = w0 * depth[tri, 0] + w1 * depth[tri, 1] + w2 * depth[tri, 2]
            flat = py * width + px
            if counter is None:
                np.maximum.at(z, flat[inside], d[inside])
            else:
                won = inside & (d >= z[flat] - 1e-9)
                np.add.at(counter, tri[won], 1)
                if owner is not None:
                    owner[flat[won]] = tri[won]

    def _index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ij = np.floor((np.asarray(xy, dtype=float) - self.origin) / self.pix).astype(np.int64)
        height, width = self.shape
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < width) & (ij[:, 1] >= 0) & (ij[:, 1] < height)
        return np.clip(ij[:, 0], 0, width - 1), np.clip(ij[:, 1], 0, height - 1), inside

    def visible(self, pts: np.ndarray, tol: float | np.ndarray, loose: bool = True) -> np.ndarray:
        """True where a view-space point is not behind the surface."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        ix, iy, inside = self._index(pts[:, :2])
        zb = (self.z_loose if loose else self.z)[iy, ix]
        return ~inside | (pts[:, 2] >= zb - tol)

    def covered(self, xy: np.ndarray) -> np.ndarray:
        """True where the part covers a view-space (x, y)."""
        ix, iy, inside = self._index(np.asarray(xy, dtype=float).reshape(-1, 2))
        return inside & np.isfinite(self.z[iy, ix])

    def poles(self, label_of_tri: np.ndarray,
              wanted: Iterable[int] | None = None) -> dict[int, tuple[np.ndarray, float]]:
        """For each label (a CAD face, or an item's faces) seen in this view:
        the visible point deepest inside its visible region, in view mm, and
        how far that point is from the region's edge (mm).

        A face's 3D centroid can fall in a hole (a ring's centroid is on the
        axis) or on a neighbour's rim; the pole of inaccessibility of what is
        actually drawn cannot. Distances are counted by erosion, which is
        exact enough to choose a label spot (needs `count=True`).
        """
        labels = np.full(self.tri_at.shape, -1, dtype=np.int64)
        seen = self.tri_at >= 0
        labels[seen] = np.asarray(label_of_tri, dtype=np.int64)[self.tri_at[seen]]
        tags = np.unique(labels[seen]) if wanted is None else np.asarray(sorted(set(wanted)))
        found: dict[int, tuple[np.ndarray, float]] = {}
        for tag in tags.tolist():
            mask = labels == tag
            if not mask.any():
                continue
            rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
            r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            region = np.pad(mask[r0:r1, c0:c1], 1)
            depth = np.zeros(region.shape, dtype=np.int64)
            current = region
            while current.any():
                depth += current
                shrunk = current.copy()
                shrunk[1:, :] &= current[:-1, :]
                shrunk[:-1, :] &= current[1:, :]
                shrunk[:, 1:] &= current[:, :-1]
                shrunk[:, :-1] &= current[:, 1:]
                current = shrunk
            i, j = np.unravel_index(int(np.argmax(depth)), depth.shape)
            centre = self.origin + (np.array([j - 1 + c0, i - 1 + r0], dtype=float) + 0.5) * self.pix
            found[int(tag)] = (centre, float(depth[i, j]) * self.pix)
        return found


def _tri_tolerance(V: np.ndarray, tris: np.ndarray, pix: float) -> np.ndarray:
    """Depth tolerance per triangle: a grazing triangle's depth changes fast
    across one pixel, so its own centroid must not be judged hidden by itself."""
    n = _normals(V, tris)
    slope = np.hypot(n[:, 0], n[:, 1]) / np.maximum(np.abs(n[:, 2]), 0.05)
    return pix * (1.0 + 1.5 * np.minimum(slope, 20.0))


def _cad_edges(mesh: Any) -> np.ndarray:
    """(E,2) node pairs along CAD curves.

    `mesh.edges` (gmsh dim-1 elements) are the real CAD edges. Without them,
    fall back to surface-triangle edges whose two triangles belong to different
    CAD faces, which is the same set for a mesh that conforms to its faces.
    """
    edges = getattr(mesh, "edges", None)
    if edges is not None:
        edges = np.asarray(edges, dtype=np.int64)
        if edges.ndim == 2 and edges.shape[1] == 2 and len(edges):
            return edges
    tris, parent = _surface(mesh)
    owner = np.asarray(mesh.boundary_face, dtype=np.int64)[parent]
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    f = np.tile(owner, 3)
    e = np.sort(e, axis=1)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, f = e[order], f[order]
    same_edge = np.all(e[1:] == e[:-1], axis=1)
    differ = same_edge & (f[1:] != f[:-1])
    return e[1:][differ]


def _chain(edges: np.ndarray) -> list[np.ndarray]:
    """Join edge segments into polylines, so a dash pattern runs along a whole
    CAD curve instead of restarting at every element."""
    if len(edges) == 0:
        return []
    e = np.unique(np.sort(np.asarray(edges, dtype=np.int64), axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, (a, b) in enumerate(e.tolist()):
        adjacency[a].append((b, i))
        adjacency[b].append((a, i))
    used = np.zeros(len(e), dtype=bool)
    lines: list[np.ndarray] = []
    starts = [n for n, nb in adjacency.items() if len(nb) != 2] + list(adjacency)
    for s in starts:
        for neighbour, edge in adjacency[s]:
            if used[edge]:
                continue
            path = [s]
            nxt = neighbour
            while True:
                used[edge] = True
                path.append(nxt)
                if len(adjacency[nxt]) != 2:
                    break
                (n1, e1), (n2, e2) = adjacency[nxt]
                nxt_node, nxt_edge = (n1, e1) if e1 != edge else (n2, e2)
                if used[nxt_edge]:
                    break
                edge, nxt = nxt_edge, nxt_node
            lines.append(np.asarray(path, dtype=np.int64))
    return lines


def _visible_runs(lines: Sequence[np.ndarray], V: np.ndarray, depth: _DepthMap,
                  step: float, tol: float) -> list[np.ndarray]:
    """The visible stretches of each polyline, as view-space (k,2) arrays.

    Each segment is cut into pieces about `step` long and each piece is kept if
    its midpoint is not behind the surface, so an edge disappearing behind a
    flange is cut where it disappears rather than at the next mesh node.
    """
    if not lines:
        return []
    a_idx = np.concatenate([ln[:-1] for ln in lines])
    b_idx = np.concatenate([ln[1:] for ln in lines])
    line_id = np.concatenate([np.full(len(ln) - 1, i) for i, ln in enumerate(lines)])
    A, B = V[a_idx], V[b_idx]
    length = np.linalg.norm(B[:, :2] - A[:, :2], axis=1)
    pieces = np.maximum(1, np.ceil(length / max(step, 1e-12))).astype(np.int64)
    seg = np.repeat(np.arange(len(A)), pieces)
    j = np.arange(int(pieces.sum())) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    t0 = (j / pieces[seg])[:, None]
    t1 = ((j + 1) / pieces[seg])[:, None]
    d = (B - A)[seg]
    p0 = A[seg] + t0 * d
    p1 = A[seg] + t1 * d
    vis = depth.visible(0.5 * (p0 + p1), tol)
    lid = line_id[seg]
    idx = np.flatnonzero(vis)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero((np.diff(idx) != 1) | (np.diff(lid[idx]) != 0)) + 1
    runs = []
    for group in np.split(idx, breaks):
        pts = np.vstack([p0[group, :2], p1[group[-1], :2][None, :]])
        runs.append(pts)
    return runs


# --------------------------------------------------------------------------
# Exact isobands
# --------------------------------------------------------------------------


def _clip(pts: np.ndarray, val: np.ndarray, cnt: np.ndarray, limit: np.ndarray,
          keep_above: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Sutherland-Hodgman step for many convex polygons at once.

    pts (m,V,D), val (m,V), cnt (m,) live vertices. Returns (m,V+1,...) with the
    survivors packed to the front. An infinite limit keeps a polygon unchanged.
    """
    m, nv, dim = pts.shape
    idx = np.arange(nv)
    valid = idx[None, :] < cnt[:, None]
    nxt = (idx[None, :] + 1) % np.maximum(cnt, 1)[:, None]
    b = np.take_along_axis(pts, nxt[..., None], axis=1)
    vb = np.take_along_axis(val, nxt, axis=1)
    lim = limit[:, None]
    if keep_above:
        ina, inb = val >= lim, vb >= lim
    else:
        ina, inb = val <= lim, vb <= lim
    crossing = (ina != inb) & valid
    with np.errstate(all="ignore"):
        t = (lim - val) / (vb - val)
    t = np.where(crossing, t, 0.0)
    cross_pts = pts + t[..., None] * (b - pts)
    cross_val = np.where(crossing, np.broadcast_to(lim, val.shape), 0.0)
    out_pts = np.stack([pts, cross_pts], axis=2).reshape(m, 2 * nv, dim)
    out_val = np.stack([val, cross_val], axis=2).reshape(m, 2 * nv)
    keep = np.stack([ina & valid, crossing], axis=2).reshape(m, 2 * nv)
    order = np.argsort(~keep, axis=1, kind="stable")[:, :nv + 1]
    return (np.take_along_axis(out_pts, order[..., None], axis=1),
            np.take_along_axis(out_val, order, axis=1),
            keep.sum(axis=1))


def _isobands(P: np.ndarray, tris: np.ndarray, s: np.ndarray,
              bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact banded contours: each linear triangle clipped into per-band pieces.

    Returns polygons (m,5,D) padded by repeating their last vertex, the band of
    each and the triangle it came from. Colouring a whole triangle by its mean
    gives the jagged, blotchy bands of a cheap plot; clipping at the band values
    with the linearly interpolated field gives the crisp isolines of Ansys.
    """
    levels = len(bounds) - 1
    band = np.clip(np.searchsorted(bounds, s, side="right") - 1, 0, levels - 1)
    tb = band[tris]
    lo, hi = tb.min(axis=1), tb.max(axis=1)
    same = lo == hi
    whole = P[tris[same]]
    whole = np.concatenate([whole, whole[:, 2:3], whole[:, 2:3]], axis=1)
    cross = np.flatnonzero(~same)
    reps = hi[cross] - lo[cross] + 1
    tri_id = np.repeat(cross, reps)
    k = np.repeat(lo[cross], reps) + (np.arange(int(reps.sum())) - np.repeat(np.cumsum(reps) - reps, reps))
    pts = P[tris[tri_id]]
    val = s[tris[tri_id]].astype(float)
    cnt = np.full(len(k), 3, dtype=np.int64)
    lower = np.where(k > 0, bounds[np.minimum(k, levels)], -np.inf)
    upper = np.where(k < levels - 1, bounds[np.minimum(k + 1, levels)], np.inf)
    pts, val, cnt = _clip(pts, val, cnt, lower, keep_above=True)
    pts, val, cnt = _clip(pts, val, cnt, upper, keep_above=False)
    ok = cnt >= 3
    pts, cnt, k, tri_id = pts[ok], cnt[ok], k[ok], tri_id[ok]
    pad = np.minimum(np.arange(5)[None, :], (cnt - 1)[:, None])
    pts = np.take_along_axis(pts, pad[..., None], axis=1)
    polys = np.concatenate([whole, pts], axis=0)
    bands = np.concatenate([lo[same], k])
    parents = np.concatenate([np.flatnonzero(same), tri_id])
    return polys, bands, parents


# --------------------------------------------------------------------------
# Canvas: one full-figure axes in pixels
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Frame:
    """View millimetres -> figure pixels: px = k * xy + offset (equal aspect)."""

    k: float
    ox: float
    oy: float

    def px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return xy[..., :2] * self.k + np.array([self.ox, self.oy])

    def mm(self, px: np.ndarray) -> np.ndarray:
        return (np.asarray(px, dtype=float) - np.array([self.ox, self.oy])) / self.k


def _fit(points: np.ndarray, target: tuple[float, float, float, float],
         margin: float = 0.03) -> _Frame:
    """Largest equal-aspect frame that puts every point inside `target` (px)."""
    pts = np.asarray(points, dtype=float)[:, :2]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    x0, y0, x1, y1 = target
    w, h = (x1 - x0) * (1 - 2 * margin), (y1 - y0) * (1 - 2 * margin)
    k = float(min(w / span[0], h / span[1]))
    centre = 0.5 * (lo + hi)
    return _Frame(k, 0.5 * (x0 + x1) - k * centre[0], 0.5 * (y0 + y1) - k * centre[1])


class _Canvas:
    """A figure whose single axes uses pixel coordinates with clipping off.

    Text is measured with the real renderer, so the layout (legend width, title
    block, label boxes) is computed from what will actually be drawn.
    """

    def __init__(self, width_in: float, height_in: float, dpi: int,
                 background: str = "gradient") -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.dpi = int(dpi)
        self.fig = Figure(figsize=(width_in, height_in), dpi=self.dpi)
        FigureCanvasAgg(self.fig)
        self.W = width_in * self.dpi
        self.H = height_in * self.dpi
        self.ax = self.fig.add_axes((0.0, 0.0, 1.0, 1.0))
        self.ax.set_xlim(0.0, self.W)
        self.ax.set_ylim(0.0, self.H)
        self.ax.set_axis_off()
        if background == "gradient":
            # Ansys' default viewport: white at the top to pale blue at the bottom.
            g = np.linspace(0.0, 1.0, 256)[:, None, None]
            top = np.array([1.0, 1.0, 1.0])
            bottom = np.array([0.78, 0.84, 0.93])
            img = bottom + (top - bottom) * g
            self.ax.imshow(np.broadcast_to(img, (256, 2, 3)), extent=(0, self.W, 0, self.H),
                           origin="lower", aspect="auto", zorder=-10, interpolation="bilinear")
        else:
            self.fig.patch.set_facecolor(background)
        self._renderer = self.fig.canvas.get_renderer()

    def inch(self, value: float) -> float:
        return value * self.dpi

    def pt(self, value: float) -> float:
        return value * self.dpi / 72.0

    def text(self, x: float, y: float, s: str, **kw: Any):  # noqa: ANN201
        artist = self.ax.text(x, y, s, **kw)
        artist.set_clip_on(False)
        return artist

    def extent(self, artist: Any) -> tuple[float, float, float, float]:
        bb = artist.get_window_extent(self._renderer)
        return (float(bb.x0), float(bb.y0), float(bb.x1), float(bb.y1))

    def measure(self, s: str, **kw: Any) -> tuple[float, float]:
        artist = self.ax.text(0, 0, s, **kw)
        x0, y0, x1, y1 = self.extent(artist)
        artist.remove()
        return x1 - x0, y1 - y0

    def save(self, out_path: Path) -> Path:
        for artist in [*self.ax.collections, *self.ax.texts, *self.ax.patches, *self.ax.lines]:
            artist.set_clip_on(False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out_path, dpi=self.dpi)
        return out_path


def _rect_overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


class _Placer:
    """Greedy label placement: near its anchor, off the legend, title, footer
    and the other labels, and (when asked) off the part so a tag never hides
    the contour it points at."""

    def __init__(self, width: float, height: float, blocked: Iterable[tuple[float, ...]],
                 coverage: Callable[[np.ndarray], np.ndarray] | None = None,
                 pad: float = 0.0, gap: float = 0.0) -> None:
        self.W, self.H = width, height
        # Fixed blocks (legend, title, footer) get breathing room, so a tag never
        # reads as one more legend entry.
        self.blocked = [(r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad) for r in blocked]
        self.gap = gap
        self.coverage = coverage

    def place(self, anchor: np.ndarray, size: tuple[float, float], *,
              radii: Sequence[float], avoid_model: float = 0.0,
              prefer: np.ndarray | None = None) -> tuple[float, float, float, float]:
        w, h = size
        best: tuple[float, tuple[float, float, float, float]] | None = None
        angles = np.radians(np.arange(0, 360, 22.5))
        sx, sy = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.15, 0.85, 3))
        for r in radii:
            for ang in ([0.0] if r == 0 else angles):
                direction = np.array([math.cos(ang), math.sin(ang)])
                c = np.asarray(anchor, dtype=float) + r * direction
                # Offset so the box's near side, not its centre, sits at radius r.
                if r > 0:
                    c = c + direction * np.array([w / 2, h / 2])
                box = (c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2)
                if box[0] < 2 or box[1] < 2 or box[2] > self.W - 2 or box[3] > self.H - 2:
                    continue
                area = w * h
                overlap = sum(_rect_overlap(box, b) for b in self.blocked) / area
                score = 50.0 * overlap + 0.006 * r
                if avoid_model and self.coverage is not None:
                    samples = np.column_stack([box[0] + sx.ravel() * w, box[1] + sy.ravel() * h])
                    score += avoid_model * float(self.coverage(samples).mean())
                if prefer is not None and r > 0:
                    score += 0.6 * (1.0 - float(direction @ prefer)) * 0.5
                if best is None or score < best[0]:
                    best = (score, box)
        if best is None:  # nowhere inside the figure: sit on the anchor
            c = np.asarray(anchor, dtype=float)
            box = (c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2)
        else:
            box = best[1]
        gap = self.gap
        self.blocked.append((box[0] - gap, box[1] - gap, box[2] + gap, box[3] + gap))
        return box


def _draw_polys(canvas: _Canvas, verts: np.ndarray, colours: np.ndarray, zorder: float = 1) -> None:
    from matplotlib.collections import PolyCollection

    pc = PolyCollection(verts, facecolors=colours, edgecolors=colours, linewidths=0.3,
                        antialiased=True, zorder=zorder)
    pc.set_clip_on(False)
    canvas.ax.add_collection(pc)


def _draw_runs(canvas: _Canvas, runs_px: Sequence[np.ndarray], colour: Any, width: float,
               zorder: float, dashed: bool = False) -> None:
    from matplotlib.collections import LineCollection

    if not runs_px:
        return
    lc = LineCollection(list(runs_px), colors=[colour], linewidths=width, zorder=zorder,
                        linestyles=[(0, (3.0, 2.2))] if dashed else ["solid"],
                        capstyle="round", joinstyle="round")
    lc.set_clip_on(False)
    canvas.ax.add_collection(lc)


def _triad(canvas: _Canvas, view: tuple[float, float], centre: tuple[float, float]) -> tuple[float, ...]:
    """XYZ triad (X red, Y green, Z blue, as Ansys draws it)."""
    from matplotlib.patches import Circle, FancyArrowPatch

    basis = _basis(*view)
    length = canvas.inch(0.42)
    cx, cy = centre
    for axis, (name, colour) in enumerate((("X", "#d62728"), ("Y", "#2ca02c"), ("Z", "#1f3fbf"))):
        unit = np.zeros(3)
        unit[axis] = 1.0
        d = basis[:2] @ unit
        tip = (cx + length * d[0], cy + length * d[1])
        arrow = FancyArrowPatch((cx, cy), tip, arrowstyle="-|>", mutation_scale=canvas.pt(9),
                                color=colour, linewidth=1.6, zorder=30)
        arrow.set_clip_on(False)
        canvas.ax.add_patch(arrow)
        norm = max(float(np.hypot(*d)), 1e-9)
        label = (cx + (length + canvas.pt(8)) * d[0] / max(norm, 0.35) * min(norm, 1.0) + 0.0,
                 cy + (length + canvas.pt(8)) * d[1] / max(norm, 0.35) * min(norm, 1.0))
        canvas.text(label[0], label[1], name, fontsize=8.5, fontweight="bold", color=colour,
                    ha="center", va="center", zorder=31)
    dot = Circle((cx, cy), canvas.pt(2.2), color="#555555", zorder=31)
    dot.set_clip_on(False)
    canvas.ax.add_patch(dot)
    pad = length + canvas.pt(14)
    return (cx - pad, cy - pad, cx + pad, cy + pad)


def _nice_length(target: float) -> float:
    if not target > 0:
        return 1.0
    exponent = math.floor(math.log10(target))
    for step in (5.0, 2.5, 2.0, 1.0):
        value = step * 10 ** exponent
        if value <= target:
            return value
    return 10 ** exponent


def _ruler(canvas: _Canvas, frame: _Frame, centre_x: float, y: float, target_px: float) -> tuple[float, ...]:
    """Scale bar in mm, exact because the projection is orthographic."""
    from matplotlib.patches import Rectangle

    length_mm = _nice_length(target_px / frame.k)
    length_px = length_mm * frame.k
    x0 = centre_x - length_px / 2
    h = canvas.pt(5)
    for i, colour in enumerate(("#000000", "#ffffff", "#000000", "#ffffff")):
        rect = Rectangle((x0 + i * length_px / 4, y), length_px / 4, h, facecolor=colour,
                         edgecolor="#000000", linewidth=0.6, zorder=30)
        rect.set_clip_on(False)
        canvas.ax.add_patch(rect)
    labels = [(x0, "0"), (x0 + length_px / 2, _num(length_mm / 2, 4)),
              (x0 + length_px, f"{_num(length_mm, 4)} (mm)")]
    for x, s in labels:
        canvas.text(x, y + h + canvas.pt(2.5), s, fontsize=8, ha="center", va="bottom",
                    color="#111111", zorder=30)
    return (x0 - canvas.pt(30), y - canvas.pt(2), x0 + length_px + canvas.pt(30), y + h + canvas.pt(16))


def _title_block(canvas: _Canvas, lines: Sequence[str], x: float, top: float) -> tuple[float, ...]:
    """Ansys' top-left block: bold first line, then one fact per line."""
    y = top
    right = x
    for i, line in enumerate(lines):
        size = 11.5 if i == 0 else 10
        t = canvas.text(x, y, line, fontsize=size, fontweight="bold" if i == 0 else "normal",
                        va="top", ha="left", color="#000000", zorder=40)
        x0, y0, x1, y1 = canvas.extent(t)
        right = max(right, x1)
        y = y0 - canvas.pt(3.0)
    return (x - canvas.pt(3), y, right + canvas.pt(4), top + canvas.pt(3))


def _legend(canvas: _Canvas, bounds: np.ndarray, x: float, top: float) -> tuple[float, ...]:
    """9 colour boxes, Max at the top, labelled at the 10 band boundaries."""
    from matplotlib.patches import Rectangle

    bh = canvas.pt(16.5)
    bw = canvas.pt(15.0)
    colours = ANSYS_LEGEND_COLORS[::-1]
    for i, colour in enumerate(colours):
        rect = Rectangle((x, top - (i + 1) * bh), bw, bh, facecolor=colour, edgecolor="#000000",
                         linewidth=0.5, zorder=40)
        rect.set_clip_on(False)
        canvas.ax.add_patch(rect)
    right = x + bw
    values = bounds[::-1]
    labels = _legend_labels(values)
    for i, label in enumerate(labels):
        tag = " Max" if i == 0 else (" Min" if i == len(values) - 1 else "")
        t = canvas.text(x + bw + canvas.pt(4), top - i * bh, f"{label}{tag}",
                        fontsize=9, va="center", ha="left", zorder=40,
                        fontweight="bold" if tag else "normal")
        right = max(right, canvas.extent(t)[2])
    return (x - canvas.pt(2), top - len(colours) * bh - canvas.pt(8), right + canvas.pt(4), top + canvas.pt(8))


def _footer(canvas: _Canvas, lines: Sequence[tuple[str, str]], x: float, bottom: float,
            max_width: float) -> tuple[float, ...]:
    """Bottom-left lines, wrapped to `max_width` px, last line lowest."""
    y = bottom
    top = bottom
    right = x
    char_w = canvas.measure("abcdefghij klmnopqrstuvwxyz 0123456789", fontsize=8.5)[0] / 37.0
    width_chars = max(30, int(max_width / max(char_w, 1e-9)))
    wrapped: list[tuple[str, str]] = []
    for text, colour in lines:
        for part in textwrap.wrap(text, width_chars) or [""]:
            wrapped.append((part, colour))
    for text, colour in reversed(wrapped):
        t = canvas.text(x, y, text, fontsize=8.5, va="bottom", ha="left", color=colour, zorder=40)
        x0, y0, x1, y1 = canvas.extent(t)
        right = max(right, x1)
        y = y1 + canvas.pt(2.0)
        top = y
    return (x - canvas.pt(2), bottom - canvas.pt(2), right + canvas.pt(2), top)


def _tag(canvas: _Canvas, box: tuple[float, ...], anchor: np.ndarray, text: str, *,
         face: str, ink: str, size: float = 9.0, leader: str = "#000000",
         marker: bool = True, hidden: bool = False) -> None:
    """A probe or letter tag: filled box with text and a leader line to its point.

    `hidden` (the point is behind the part from this camera) draws a dashed
    leader and a hollow marker, so the tag does not pretend to sit on the
    surface it is drawn over."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax_, ay = float(anchor[0]), float(anchor[1])
    # Leader from the nearest point of the box edge to the anchor.
    ex = min(max(ax_, x0), x1)
    ey = min(max(ay, y0), y1)
    if math.hypot(ex - ax_, ey - ay) > canvas.pt(2):
        line = Line2D([ex, ax_], [ey, ay], color=leader, linewidth=0.9, zorder=48,
                      linestyle=(0, (2.5, 1.8)) if hidden else "-")
        line.set_clip_on(False)
        canvas.ax.add_line(line)
    if marker:
        dot = Line2D([ax_], [ay], marker="o", markersize=3.6,
                     markerfacecolor="#ffffff" if hidden else face,
                     markeredgecolor="#000000", markeredgewidth=0.6, zorder=49)
        dot.set_clip_on(False)
        canvas.ax.add_line(dot)
    patch = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="square,pad=0",
                           facecolor=face, edgecolor="#000000", linewidth=0.7, zorder=50)
    patch.set_clip_on(False)
    canvas.ax.add_patch(patch)
    canvas.text(cx, cy, text, fontsize=size, fontweight="bold", color=ink,
                ha="center", va="center", zorder=51)


def _tag_size(canvas: _Canvas, text: str, size: float = 9.0) -> tuple[float, float]:
    w, h = canvas.measure(text, fontsize=size, fontweight="bold")
    return w + canvas.pt(7), h + canvas.pt(4)


def _layout_target(canvas: _Canvas, points_view: np.ndarray, title: tuple[float, ...],
                   left: float, bottom: float, top: float) -> _Frame:
    """Fit the model right of the legend and above the footer; if it would run
    under the title block, take the better of 'below the title' and 'right of
    the title'. The model is never drawn under text."""
    g = canvas.inch(0.15)
    full = (left + g, bottom + g, canvas.W - g, top - g)
    frame = _fit(points_view, full)
    px = frame.px(points_view)
    inside = ((px[:, 0] >= title[0]) & (px[:, 0] <= title[2])
              & (px[:, 1] >= title[1]) & (px[:, 1] <= title[3]))
    if not inside.any():
        return frame
    below = _fit(points_view, (left + g, bottom + g, canvas.W - g, title[1] - g))
    beside = _fit(points_view, (title[2] + g, bottom + g, canvas.W - g, top - g))
    return below if below.k >= beside.k else beside


# --------------------------------------------------------------------------
# Camera choice
# --------------------------------------------------------------------------


def _case_face_weights(case: Any) -> list[tuple[float, tuple[int, ...]]]:
    items: list[tuple[float, tuple[int, ...]]] = []
    for support in getattr(case, "supports", ()) or ():
        items.append((1.0, tuple(int(t) for t in support.faces)))
    for load in getattr(case, "loads", ()) or ():
        items.append((2.0, tuple(int(t) for t in load.faces)))
    return items


#: Eight iso views: the default and its quarter turns, from above and below.
_CANDIDATE_VIEWS: tuple[tuple[float, float], ...] = tuple(
    (elev, ((DEFAULT_VIEW[1] + turn + 180.0) % 360.0) - 180.0)
    for elev in (DEFAULT_VIEW[0], -DEFAULT_VIEW[0]) for turn in (0.0, 90.0, 180.0, 270.0)
)


@dataclass(frozen=True)
class _ViewScore:
    view: tuple[float, float]
    seen_items: int            # lettered items with >= 10% of their area visible
    score: float               # weighted, saturating visible fraction
    probes: tuple[bool, ...]   # is each probe node visible from here
    peak: float                # largest `field` value on a visible surface node


def _score_views(mesh: Any, case: Any, probe_nodes: Sequence[int] = (),
                 field: np.ndarray | None = None) -> list[_ViewScore]:
    """Visible area of the case's faces, and visibility of given nodes, from
    each candidate view, with the same z-buffer that hides edges."""
    items = _case_face_weights(case) if case is not None else []
    nodes = np.asarray(mesh.nodes, dtype=float)
    tris, parent = _surface(mesh)
    owner = np.asarray(mesh.boundary_face, dtype=np.int64)[parent]
    area3 = 0.5 * np.linalg.norm(np.cross(nodes[tris[:, 1]] - nodes[tris[:, 0]],
                                          nodes[tris[:, 2]] - nodes[tris[:, 0]]), axis=1)
    masks = [(weight, np.isin(owner, faces)) for weight, faces in items]
    probe_nodes = [int(n) for n in probe_nodes]
    rows = []
    for view in _CANDIDATE_VIEWS:
        V = _project(nodes, view)
        front = _area2(V, tris) > 0
        depth = _DepthMap(V, tris[front], resolution=220, count=bool(masks))
        visible_area = np.zeros(len(tris))
        if masks:
            visible_area[front] = depth.tri_pixels * depth.pix ** 2
        seen_items = 0
        score = 0.0
        for weight, mask in masks:
            total = float(area3[mask].sum())
            seen = float(visible_area[mask].sum())
            if total > 0 and seen >= 0.1 * total:
                seen_items += 1
            if total > 0:
                # Saturates: a face seen at half its area or more counts fully,
                # so a view is not preferred merely for looking more squarely.
                score += weight * min(1.0, 2.0 * seen / total)
        if view[0] < 0:
            score *= 0.8
        probes = tuple(bool(v) for v in depth.visible(V[probe_nodes], 3.0 * depth.pix)) \
            if probe_nodes else ()
        peak = -np.inf
        if field is not None:
            shown = np.unique(tris[front])
            shown = shown[depth.visible(V[shown], 3.0 * depth.pix)]
            peak = float(field[shown].max()) if len(shown) else -np.inf
        rows.append(_ViewScore(view, seen_items, score, probes, peak))
    return rows


def _pick_case_view(rows: Sequence[_ViewScore]) -> tuple[float, float]:
    """Best (seen items, score); the default wins unless clearly beaten."""
    default = rows[0]
    best = max(rows, key=lambda r: (r.seen_items, r.score))  # first maximum: default first
    if best is not default and best.seen_items == default.seen_items \
            and best.score < 1.25 * default.score + 1e-12:
        return default.view
    return best.view


def choose_view(mesh: Any, case: Any = None) -> tuple[float, float]:
    """(elevation, azimuth) for the pictures: the default iso view unless
    another iso view shows the loaded and fixed faces clearly better.

    Ansys' default iso view often hides the face a load is on (the far end of
    a cantilever, the underside of a shelf). Each of eight iso views (the
    default and its quarter turns, from above and below) is scored by how many
    lettered items show at least a tenth of their area, then by the visible
    fraction of their faces (loads weighted twice supports, saturating at half
    the area), using the same z-buffer that hides edges. Views from above are
    preferred, and the default is kept unless something is clearly better, so
    similar parts get the same camera.
    """
    if case is None or not _case_face_weights(case):
        return DEFAULT_VIEW
    return _pick_case_view(_score_views(mesh, case))


def _result_view(solution: Any) -> tuple[float, float]:
    """The camera for result pictures: `choose_view`'s, unless it hides the
    Max Equivalent Stress and another view from above shows it.

    A result picture whose Max tag points behind the part makes the reader take
    the hotspot on trust, so a view from above that shows the Max node wins
    (preferring one that also shows the value away from a singularity, the Max
    deformation and the most of the case). Looking at a part from below is
    unnatural, so it is used only when no view from above shows the Max and
    the hottest place every view from above shows is clearly cooler (< 95 %)
    than the Max -- the underside of a loaded shelf next to a top surface just
    as hot keeps a natural view (the case view if it shows that hot surface,
    else the view from above that does), with the hidden probe drawn dashed.
    Both result
    pictures share the camera, as in Ansys, so they compare side by side.
    """
    mesh, case = solution.mesh, getattr(solution, "case", None)
    field = np.asarray(solution.von_mises_mpa, dtype=float)
    imax, _, _ = _hotspot(solution, "max_stress", field, np.argmax)
    probes = [imax]
    away = getattr(solution, "max_stress_away", None)
    if away is not None and getattr(away, "node", None) is not None:
        probes.append(int(away.node))
    deformation = getattr(solution, "total_deformation_mm", None)
    if deformation is not None:
        probes.append(_hotspot(solution, "max_deformation", np.asarray(deformation, float), np.argmax)[0])
    rows = _score_views(mesh, case, probes, field)
    base = _pick_case_view(rows) if case is not None and _case_face_weights(case) else DEFAULT_VIEW
    base_row = next(r for r in rows if r.view == base)
    if base_row.probes[0]:
        return base

    def best(candidates: list[_ViewScore]) -> tuple[float, float]:
        # Most probes shown, then most of the case; near-ties (within 5 %) go
        # to the earliest candidate, i.e. the default view and its neighbours.
        top = max((sum(r.probes), r.seen_items) for r in candidates)
        tied = [r for r in candidates if (sum(r.probes), r.seen_items) == top]
        high = max(r.score for r in tied)
        return next(r for r in tied if r.score >= 0.95 * high).view

    hot = 0.95 * float(field[imax])
    above = [r for r in rows if r.view[0] > 0 and r.probes[0]]
    if above:
        return best(above)
    if base_row.peak >= hot:
        return base
    warm = [r for r in rows if r.view[0] > 0 and r.peak >= hot]
    if warm:
        return best(warm)
    below = [r for r in rows if r.probes[0]]
    return best(below) if below else base


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


def _element_description(mesh: Any) -> str:
    order = int(getattr(mesh, "order", 2))
    text = "10-node quadratic tetrahedra" if order == 2 else "4-node linear tetrahedra"
    if getattr(mesh, "straight_sided", False):
        text += " (straight-sided)"
    return text


def _material_phrase(case: Any) -> str:
    material = getattr(case, "material", None)
    if material is None:
        return "material not recorded"
    return (f"{material.name} (E {_num(material.youngs_modulus_mpa, 6)} MPa, "
            f"ν {_num(material.poisson_ratio, 3)})")


def _hotspot(solution: Any, attr: str, field: np.ndarray, pick: Callable[[np.ndarray], int]) -> tuple[int, bool, str | None]:
    spot = getattr(solution, attr, None)
    if spot is not None and getattr(spot, "node", None) is not None:
        node = int(spot.node)
        if 0 <= node < len(field):
            return node, bool(getattr(spot, "singular", False)), getattr(spot, "reason", None)
    return int(pick(field)), False, None


def _scale_factor(solution: Any, scale: Any, displacement: np.ndarray, nodes: np.ndarray) -> tuple[float, str]:
    if isinstance(scale, str):
        key = scale.strip().lower()
        if key == "auto":
            auto = getattr(solution, "deformation_scale_auto", None)
            if auto is None or not math.isfinite(float(auto)) or float(auto) <= 0:
                umax = float(np.linalg.norm(displacement, axis=1).max()) if len(displacement) else 0.0
                extent = float((nodes.max(axis=0) - nodes.min(axis=0)).max())
                auto = 0.05 * extent / umax if umax > 0 else 1.0
            return float(auto), "Auto Scale"
        if key == "true":
            return 1.0, "True Scale"
        raise StructuralRenderError(
            f"scale must be 'auto', 'true' or a number, not {scale!r}"
        )
    if isinstance(scale, bool):
        raise StructuralRenderError(f"scale must be 'auto', 'true' or a number, not {scale!r}")
    try:
        value = float(scale)
    except (TypeError, ValueError):
        raise StructuralRenderError(
            f"scale must be 'auto', 'true' or a number, not {scale!r}"
        ) from None
    if not math.isfinite(value) or value < 0:
        raise StructuralRenderError(f"scale must be a finite number >= 0, not {scale!r}")
    return value, "True Scale" if value == 1.0 else "User Defined"


def _field(solution: Any, result: _Result, n_nodes: int) -> np.ndarray:
    if result.key == "equivalent_stress":
        values = getattr(solution, "von_mises_mpa", None)
    else:
        values = getattr(solution, "total_deformation_mm", None)
        if values is None and getattr(solution, "displacement_mm", None) is not None:
            values = np.linalg.norm(np.asarray(solution.displacement_mm, dtype=float), axis=1)
    if values is None:
        raise StructuralRenderError(f"the solution carries no {result.title} field to draw")
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape[0] != n_nodes:
        raise StructuralRenderError(
            f"{result.title} has {values.shape[0]} values but the mesh has {n_nodes} nodes; "
            "it needs one value per node"
        )
    if not np.all(np.isfinite(values)):
        raise StructuralRenderError(
            f"{result.title} is not finite at {int((~np.isfinite(values)).sum())} nodes; "
            "the solve failed and must not be drawn as a result"
        )
    return values


def render_result(solution: Any, quantity: str, out_path: str | Path, *,
                  scale: str | float = "auto", view: Sequence[float] | None = None,
                  show_undeformed: bool = True, dpi: int = 150) -> Path:
    """Draw Equivalent Stress or Total Deformation the way Ansys Mechanical does.

    The contour sits on the deformed shape (nodes + scale x displacement), with
    an exact 9-band legend running Min -> Max, Max/Min probes at the hotspot
    nodes, the undeformed CAD edges dashed behind it and the title block,
    triad, ruler and footer an Ansys user reads without thinking.

    Honesty rules carried into the pixels: the header says SOLVED and names the
    element; the stress Max tag says "(singular)" when the solver flagged it, and
    the footer then says it rises with refinement and gives the value away from
    the singularity. Nothing from the surrogate is ever drawn here.

    `quantity` is "equivalent_stress" or "total_deformation"; `scale` is "auto"
    (the solver's /DSCALE-style auto factor), "true" (1.0) or a number. The
    format follows the suffix of `out_path` (.png, .svg, .pdf).
    """
    import time

    started = time.perf_counter()
    if quantity not in _RESULTS:
        raise StructuralRenderError(
            f"quantity must be one of {', '.join(QUANTITIES)}, not {quantity!r}"
        )
    result = _RESULTS[quantity]
    out_path = Path(out_path)
    mesh = solution.mesh
    case = getattr(solution, "case", None)
    nodes = np.asarray(mesh.nodes, dtype=float)
    displacement = np.asarray(solution.displacement_mm, dtype=float)
    if displacement.shape != nodes.shape:
        raise StructuralRenderError(
            f"displacement has shape {displacement.shape} but the mesh nodes are {nodes.shape}"
        )
    if not np.all(np.isfinite(displacement)):
        raise StructuralRenderError("displacement is not finite; the solve failed and must not be drawn")
    field = _field(solution, result, len(nodes))
    factor, scale_kind = _scale_factor(solution, scale, displacement, nodes)
    view = _check_view(view) if view is not None else _result_view(solution)

    deformed = nodes + factor * displacement
    V = _project(deformed, view)
    tris, parent = _surface(mesh)
    owner = np.asarray(mesh.boundary_face, dtype=np.int64)[parent]
    a2 = _area2(V, tris)
    front = a2 > 0
    ftris = tris[front]
    depth = _DepthMap(V, ftris)

    vmin, vmax = float(field.min()), float(field.max())
    flat = not (vmax - vmin > 1e-12 * max(abs(vmax), abs(vmin), 1e-30))
    bounds = np.linspace(vmin, vmax, LEGEND_LEVELS + 1)
    if flat:
        polys = np.concatenate([V[ftris], V[ftris][:, 2:3], V[ftris][:, 2:3]], axis=1)
        bands = np.zeros(len(ftris), dtype=np.int64)
        parents = np.arange(len(ftris))
    else:
        polys, bands, parents = _isobands(V, ftris, field, bounds)
    from matplotlib.colors import to_rgba_array

    palette = to_rgba_array(ANSYS_LEGEND_COLORS)
    normals = _smooth_normals(V, ftris, owner[front])
    colours = _shade(palette[bands], normals[parents])
    nearness = V[ftris][..., 2].mean(axis=1)[parents]
    order = np.argsort(nearness, kind="stable")

    canvas = _Canvas(10.0, 7.0, dpi)
    margin = canvas.inch(0.16)
    title_lines = [
        "A: Static Structural",
        result.title,
        f"Type: {result.type_line}",
        f"Unit: {result.unit}",
        "Time: 1 s",
        f"Deformation scale: {_num(factor, 3)} ({scale_kind})",
    ]
    title = _title_block(canvas, title_lines, margin, canvas.H - margin)
    legend = _legend(canvas, bounds, margin + canvas.pt(2), title[1] - canvas.pt(12))
    header = canvas.text(canvas.W - margin, canvas.H - margin, structural_header(int(getattr(mesh, "order", 2))),
                         fontsize=8.5, ha="right", va="top", color="#333333", zorder=40)
    header_box = canvas.extent(header)

    node_count = int(getattr(mesh, "node_count", len(nodes)))
    element_count = int(getattr(mesh, "element_count", len(np.asarray(mesh.elements))))
    footer_lines = [(f"{node_count:,} nodes · {element_count:,} elements · "
                     f"{_element_description(mesh)} · {_material_phrase(case)}", "#222222")]
    imax, max_singular, max_reason = _hotspot(
        solution, "max_stress" if result.key == "equivalent_stress" else "max_deformation", field, np.argmax)
    imin, _, _ = _hotspot(
        solution, "min_stress" if result.key == "equivalent_stress" else "min_deformation", field, np.argmin)
    away = getattr(solution, "max_stress_away", None) if result.key == "equivalent_stress" else None
    max_singular = max_singular and result.key == "equivalent_stress"
    if max_singular:
        sentence = "Max is at or next to a singularity"
        if max_reason:
            sentence += f" ({max_reason})"
        sentence += " and rises with mesh refinement"
        if away is not None and getattr(away, "value", None) is not None:
            radius = getattr(solution, "exclusion_radius_mm", None)
            beyond = f" (beyond {_num(float(radius), 3)} mm)" if radius else ""
            sentence += f"; away from it{beyond}: {_num(float(away.value), 4)} MPa"
        footer_lines.append((sentence, "#8a1c1c"))
    footer = _footer(canvas, footer_lines, margin, margin * 0.7, canvas.W - 2 * margin)

    # Model: right of the legend, above the ruler/triad strip, below the header.
    strip_top = footer[3] + canvas.inch(0.62)
    undeformed_V = _project(nodes, view) if show_undeformed and factor > 0 else None
    fit_points = V[np.unique(tris)]
    if undeformed_V is not None:
        fit_points = np.vstack([fit_points, undeformed_V[np.unique(tris)]])
    frame = _layout_target(canvas, fit_points, title, legend[2], strip_top,
                           min(header_box[1], canvas.H - margin))

    verts = frame.px(polys[order][..., :2])
    _draw_polys(canvas, verts, colours[order], zorder=1)

    lines = _chain(_cad_edges(mesh))
    step = 2.0 / frame.k
    tol = 2.0 * depth.pix
    if undeformed_V is not None:
        runs = _visible_runs(lines, undeformed_V, depth, step, tol * 1.5)
        _draw_runs(canvas, [frame.px(r) for r in runs], (0.25, 0.25, 0.25, 0.75), 0.7,
                   zorder=2, dashed=True)
    runs = _visible_runs(lines, V, depth, step, tol)
    _draw_runs(canvas, [frame.px(r) for r in runs], (0.0, 0.0, 0.0, 0.9), 0.6, zorder=3)

    # Ruler (bottom centre) and triad (bottom right), in the strip above the footer.
    triad_box = _triad(canvas, view, (canvas.W - canvas.inch(0.75), footer[3] + canvas.inch(0.36)))
    ruler_box = _ruler(canvas, frame, canvas.W * 0.55, footer[3] + canvas.inch(0.2), canvas.W * 0.2)

    def coverage(px: np.ndarray) -> np.ndarray:
        return depth.covered(frame.mm(px))

    placer = _Placer(canvas.W, canvas.H, [title, legend, header_box, footer, triad_box, ruler_box],
                     coverage, pad=canvas.pt(14), gap=canvas.pt(12))
    centre_px = frame.px(V[np.unique(tris)][:, :2]).mean(axis=0)
    probes = [(imax, "Max", ["singular"] if max_singular else [], "#ff0000", "#000000")]
    probes.append((imin, "Min", [], "#0000ff", "#ffffff"))
    if max_singular and away is not None and getattr(away, "node", None) is not None:
        node = int(away.node)
        if 0 <= node < len(nodes) and node != imax:
            probes.append((node, f"Max away from singularity: {_num(float(away.value), 4)}", [],
                           "#ffb200", "#000000"))
    radii = [canvas.pt(r) for r in (18, 30, 46, 64, 90, 120, 160, 210)]
    # Reserve every probe point first so no tag lands on another probe's dot,
    # and the hottest visible contour too: a tag must not cover the result it
    # names (two tags once sat on a beam's red root band).
    for node, *_ in probes:
        p = frame.px(V[node, :2])
        placer.blocked.append((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4))
    shown = np.unique(ftris)
    hot = shown[field[shown] >= vmin + 0.6 * (vmax - vmin)] if not flat else shown[:0]
    hot = hot[depth.visible(V[hot], 3.0 * depth.pix)] if len(hot) else hot
    if len(hot) > 600:
        hot = hot[np.linspace(0, len(hot) - 1, 600).astype(np.int64)]
    for q in frame.px(V[hot][:, :2]) if len(hot) else ():
        placer.blocked.append((q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5))
    for node, name, notes, face, ink in probes:
        anchor = frame.px(V[node, :2])
        seen = bool(depth.visible(V[node][None, :], 3.0 * depth.pix)[0])
        # A probe on the far side says so in words, as the setup's "A (behind)"
        # does; the dashed leader alone was easy to miss.
        notes = notes + ([] if seen else ["behind"])
        text = f"{name} ({', '.join(notes)})" if notes else name
        direction = anchor - centre_px
        norm = float(np.linalg.norm(direction))
        prefer = direction / norm if norm > 1e-9 else np.array([0.0, 1.0])
        size = _tag_size(canvas, text)
        box = placer.place(anchor, size, radii=radii, avoid_model=3.0, prefer=prefer)
        _tag(canvas, box, anchor, text, face=face, ink=ink, hidden=not seen)

    path = canvas.save(out_path)
    _ = time.perf_counter() - started
    return path


# --------------------------------------------------------------------------
# Setup (environment) picture
# --------------------------------------------------------------------------


def _item_title(item: Any, kind: str) -> str:
    label = getattr(item, "label", kind)
    return label if label == kind else f"{label} ({kind})"


def _sample_points(centroids: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    """Up to `count` well-spread triangle indices (farthest-point sampling),
    starting from the one nearest the area centroid."""
    if len(centroids) == 0:
        return np.zeros(0, dtype=np.int64)
    centre = (centroids * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-300)
    chosen = [int(np.argmin(np.linalg.norm(centroids - centre, axis=1)))]
    dist = np.linalg.norm(centroids - centroids[chosen[0]], axis=1)
    spread = float(dist.max())
    while len(chosen) < min(count, len(centroids)):
        nxt = int(np.argmax(dist))
        if dist[nxt] < 0.18 * spread or dist[nxt] <= 0:
            break
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(centroids - centroids[nxt], axis=1))
    return np.asarray(chosen, dtype=np.int64)


def render_setup(mesh: Any, case: Any, out_path: str | Path, *,
                 view: Sequence[float] | None = None, dpi: int = 150) -> Path:
    """The INPUT picture: the Ansys environment view of supports and loads.

    Fixed supports are painted blue-violet, each load in its own red-family
    colour with arrows (a force along its vector, a pressure normal into its
    faces), gravity as one gold arrow, and every item tagged with the letter
    the confirmation box uses. No result colours appear here -- this is what
    was asked, not what came out.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyArrowPatch, Rectangle

    out_path = Path(out_path)
    view = _check_view(view) if view is not None else choose_view(mesh, case)
    nodes = np.asarray(mesh.nodes, dtype=float)
    tris, parent = _surface(mesh)
    owner = np.asarray(mesh.boundary_face, dtype=np.int64)[parent]
    known = set(int(t) for t in np.unique(owner))
    lettered = list(case.lettered()) if hasattr(case, "lettered") else []
    for letter, item in lettered:
        faces = getattr(item, "faces", None)
        if faces is not None:
            missing = sorted(set(int(t) for t in faces) - known)
            if missing:
                raise StructuralRenderError(
                    f"{letter} ({item.label}) names face(s) {missing}, which this mesh does not "
                    f"have; its faces are {sorted(known)}"
                )

    V = _project(nodes, view)
    a2 = _area2(V, tris)
    front = a2 > 0
    ftris = tris[front]
    fowner = owner[front]
    depth = _DepthMap(V, ftris, count=True)

    # Colour per item; loads drawn over supports where a face carries both.
    item_colour: dict[str, str] = {}
    face_colour: dict[int, str] = {}
    load_index = 0
    kinds: dict[str, str] = {}
    for letter, item in lettered:
        kind = type(item).__name__
        kinds[letter] = kind
        if kind == "FixedSupport":
            item_colour[letter] = SUPPORT_COLOR
        elif kind == "Gravity":
            item_colour[letter] = GRAVITY_COLOR
        else:
            item_colour[letter] = LOAD_COLORS[load_index % len(LOAD_COLORS)]
            load_index += 1
    for letter, item in lettered:
        if kinds[letter] == "FixedSupport":
            for t in item.faces:
                face_colour[int(t)] = item_colour[letter]
    for letter, item in lettered:
        if kinds[letter] in ("ForceLoad", "PressureLoad"):
            for t in item.faces:
                face_colour[int(t)] = item_colour[letter]

    base = np.tile(to_rgba(PART_GREY), (len(ftris), 1))
    for tag, colour in face_colour.items():
        base[fowner == tag] = to_rgba(colour)
    colours = _shade(base, _smooth_normals(V, ftris, fowner))
    order = np.argsort(V[ftris][..., 2].mean(axis=1), kind="stable")

    diag = float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0)))
    arrow_mm = 0.11 * diag
    tri_centroid = nodes[tris].mean(axis=1)
    tri_area = 0.5 * np.linalg.norm(np.cross(nodes[tris[:, 1]] - nodes[tris[:, 0]],
                                             nodes[tris[:, 2]] - nodes[tris[:, 0]]), axis=1)
    tri_normal = _normals(nodes, tris)
    visible_tri = np.zeros(len(tris), dtype=bool)
    tol = _tri_tolerance(V, ftris, depth.pix)
    cen_v = V[ftris].mean(axis=1)
    visible_tri[np.flatnonzero(front)] = (depth.tri_pixels > 0) | depth.visible(cen_v, tol, loose=False)
    # Stricter, for arrows: the triangle's own centre is in view, not just a
    # pixel of it at the edge of an occluder.
    centre_seen = np.zeros(len(tris), dtype=bool)
    centre_seen[np.flatnonzero(front)] = depth.visible(cen_v, tol, loose=False)
    surface_v = V[np.unique(tris)][:, :2]
    model_span = max(float(np.ptp(surface_v, axis=0).max()), 1e-9)

    def shaft_seen(tail: np.ndarray, head: np.ndarray) -> bool:
        """No part of the arrow passes behind the part (a bore's far-side arrow
        crossing the fixed face in front of it read as a load on that face)."""
        ends = _project(np.vstack([tail, head]), view)
        steps = np.linspace(0.0, 1.0, 7)[:, None]
        points = ends[0] + steps * (ends[1] - ends[0])
        return bool(depth.visible(points, 2.5 * depth.pix, loose=True).all())

    arrows: list[tuple[np.ndarray, np.ndarray, str]] = []   # (tail3d, head3d, colour)
    # letter -> (view-space anchor (x, y), visible)
    anchors: dict[str, tuple[np.ndarray | None, bool]] = {}
    for letter, item in lettered:
        kind = kinds[letter]
        if kind == "Gravity":
            continue
        mask = np.isin(owner, [int(t) for t in item.faces])
        vis_mask = mask & visible_tri
        pool = np.flatnonzero(vis_mask if vis_mask.any() else mask)
        weights = tri_area[pool]
        centroid_pool = tri_centroid[pool]
        # Label anchor: the visible point deepest inside the item's visible
        # faces (a bore's centroid is on its axis, in empty space); hidden
        # items fall back to the triangle nearest their area centroid.
        pole = depth.poles(np.isin(owner[front], [int(t) for t in item.faces]).astype(np.int64),
                           wanted=[1]).get(1)
        if pole is not None:
            anchors[letter] = (pole[0], True)
        else:
            centre = (centroid_pool * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-300)
            anchor_tri = pool[int(np.argmin(np.linalg.norm(centroid_pool - centre, axis=1)))]
            anchors[letter] = (_project(tri_centroid[anchor_tri][None, :], view)[0, :2], False)
        if kind == "FixedSupport":
            continue
        # As many arrows as the face has room for on screen: 9 on a small end
        # face overlap into one blob of arrowheads.
        strict = np.flatnonzero(mask & centre_seen)
        pool = strict if strict.size else pool
        span = float(np.ptp(V[np.unique(tris[pool])][:, :2], axis=0).max()) / model_span
        wanted = 1 if span < 0.03 else 3 if span < 0.09 else 9
        picks = pool[_sample_points(tri_centroid[pool], tri_area[pool], 3 * wanted)]
        made = 0
        for t in picks:
            if made >= wanted:
                break
            p = tri_centroid[t]
            n = tri_normal[t]
            if kind == "ForceLoad":
                d = np.asarray(item.direction, dtype=float)
                if float(d @ n) < 0:     # pushes into the face: head on the face
                    arrow = (p - d * arrow_mm, p, item_colour[letter])
                else:                    # pulls: tail on the face
                    arrow = (p, p + d * arrow_mm, item_colour[letter])
            else:
                if float(item.pressure_mpa) > 0:
                    arrow = (p + n * arrow_mm, p, item_colour[letter])
                else:
                    arrow = (p, p + n * arrow_mm, item_colour[letter])
            if strict.size and not shaft_seen(arrow[0], arrow[1]):
                continue
            arrows.append(arrow)
            made += 1

    surface_nodes = np.unique(tris)
    fit_points = [V[surface_nodes]]
    for tail, head, _ in arrows:
        fit_points.append(_project(np.vstack([tail, head]), view))
    gravity_arrow = None
    for letter, item in lettered:
        if kinds[letter] == "Gravity":
            lo, hi = V[surface_nodes].min(axis=0), V[surface_nodes].max(axis=0)
            span = float((hi - lo)[:2].max())
            d = _basis(*view) @ np.asarray(item.direction, dtype=float)
            length = 0.22 * span
            tail = np.array([hi[0] + 0.12 * span, hi[1] - 0.05 * span, hi[2]])
            if d[1] > 0:
                tail[1] = lo[1] + 0.25 * span
            head = tail + np.array([d[0], d[1], 0.0]) * length
            gravity_arrow = (letter, tail, head)
            fit_points.append(np.vstack([tail, head]))
    fit_points = np.vstack(fit_points)

    canvas = _Canvas(10.0, 7.0, dpi)
    margin = canvas.inch(0.16)
    title = _title_block(canvas, ["A: Static Structural", "Static Structural", "Time: 1 s"],
                         margin, canvas.H - margin)
    # Item legend: letter box in the item colour, then its description.
    y = title[1] - canvas.pt(10)
    right = title[2]
    box = canvas.pt(14)
    for letter, item in lettered:
        kind = kinds[letter]
        if kind == "FixedSupport":
            text = f"{_item_title(item, 'Fixed Support')} — {_faces_phrase(item.faces)}"
        elif kind == "ForceLoad":
            text = (f"{_item_title(item, 'Force')}: {_num(item.magnitude_n, 4)} N along "
                    f"{_direction_name(item.vector_n)} — {_faces_phrase(item.faces)}")
        elif kind == "PressureLoad":
            text = f"{_item_title(item, 'Pressure')}: {_num(item.pressure_mpa, 4)} MPa — {_faces_phrase(item.faces)}"
        else:
            text = (f"{_item_title(item, 'Standard Earth Gravity')}: {_num(item.magnitude_mps2, 5)} m/s² "
                    f"along {_direction_name(item.acceleration_mps2)}")
        rect = Rectangle((margin, y - box), box * max(1, len(letter)), box,
                         facecolor=item_colour[letter], edgecolor="#000000", linewidth=0.6, zorder=40)
        rect.set_clip_on(False)
        canvas.ax.add_patch(rect)
        canvas.text(margin + box * max(1, len(letter)) / 2, y - box / 2, letter, fontsize=8.5,
                    fontweight="bold", color="#ffffff", ha="center", va="center", zorder=41)
        t = canvas.text(margin + box * max(1, len(letter)) + canvas.pt(5), y - box / 2, text,
                        fontsize=9.5, ha="left", va="center", zorder=40)
        right = max(right, canvas.extent(t)[2])
        y -= box + canvas.pt(5)
    legend = (margin - canvas.pt(3), y, right + canvas.pt(4), title[1])
    header = canvas.text(canvas.W - margin, canvas.H - margin,
                         "Static Structural setup — inputs, not results",
                         fontsize=8.5, ha="right", va="top", color="#333333", zorder=40)
    header_box = canvas.extent(header)
    node_count = int(getattr(mesh, "node_count", len(nodes)))
    element_count = int(getattr(mesh, "element_count", len(np.asarray(mesh.elements))))
    face_count = len(getattr(mesh, "faces", ()) or known)
    part = getattr(mesh, "path", None)
    footer_text = (f"{Path(part).name + ' · ' if part else ''}{face_count} faces · "
                   f"{node_count:,} nodes · {element_count:,} elements · "
                   f"{_element_description(mesh)} · {_material_phrase(case)}")
    footer = _footer(canvas, [(footer_text, "#222222")], margin, margin * 0.7, canvas.W - 2 * margin)
    strip_top = footer[3] + canvas.inch(0.62)
    # The item legend can be wide: keep the model right of the title only, and
    # let the layout move it below the legend if they would overlap.
    left_block = (title[0], legend[1], max(title[2], legend[2]), title[3])
    frame = _layout_target(canvas, fit_points, left_block, margin, strip_top,
                           min(header_box[1], canvas.H - margin))

    verts = frame.px(V[ftris][order][..., :2])
    _draw_polys(canvas, verts, colours[order], zorder=1)
    cad_edges = _cad_edges(mesh)
    lines = _chain(cad_edges)
    runs = _visible_runs(lines, V, depth, 2.0 / frame.k, 2.0 * depth.pix)
    _draw_runs(canvas, [frame.px(r) for r in runs], (0.1, 0.1, 0.1, 0.9), 0.7, zorder=3)
    # An item on faces the camera cannot see is outlined through the part, dashed
    # in its colour, so "A (behind)" still points at a place.
    for letter, item in lettered:
        point, visible = anchors.get(letter, (None, True))
        if visible or point is None:
            continue
        on_item = np.zeros(len(nodes), dtype=bool)
        on_item[np.unique(tris[np.isin(owner, [int(t) for t in item.faces])])] = True
        outline = cad_edges[on_item[cad_edges[:, 0]] & on_item[cad_edges[:, 1]]]
        hidden_runs = [V[line][:, :2] for line in _chain(outline)]
        _draw_runs(canvas, [frame.px(r) for r in hidden_runs], item_colour[letter], 1.5,
                   zorder=4, dashed=True)

    from matplotlib import patheffects

    outline = [patheffects.withStroke(linewidth=3.4, foreground="#ffffff")]
    for tail, head, colour in arrows:
        t2, h2 = frame.px(_project(np.vstack([tail, head]), view)[:, :2])
        # Darker than the face colour it is drawn over, and outlined, so a red
        # arrow on a red loaded face still reads.
        ink = tuple(0.62 * c for c in to_rgba(colour)[:3])
        arrow = FancyArrowPatch(tuple(t2), tuple(h2), arrowstyle="-|>", mutation_scale=canvas.pt(10),
                                color=ink, linewidth=1.6, zorder=20, shrinkA=0, shrinkB=0,
                                path_effects=outline)
        arrow.set_clip_on(False)
        canvas.ax.add_patch(arrow)
    if gravity_arrow is not None:
        letter, tail, head = gravity_arrow
        t2, h2 = frame.px(np.vstack([tail[:2], head[:2]]))
        arrow = FancyArrowPatch(tuple(t2), tuple(h2), arrowstyle="-|>", mutation_scale=canvas.pt(13),
                                color=GRAVITY_COLOR, linewidth=2.6, zorder=20, shrinkA=0, shrinkB=0)
        arrow.set_clip_on(False)
        canvas.ax.add_patch(arrow)
        anchors[letter] = (None, True)  # type: ignore[assignment]

    triad_box = _triad(canvas, view, (canvas.W - canvas.inch(0.75), footer[3] + canvas.inch(0.36)))
    ruler_box = _ruler(canvas, frame, canvas.W * 0.55, footer[3] + canvas.inch(0.2), canvas.W * 0.2)

    def coverage(px: np.ndarray) -> np.ndarray:
        return depth.covered(frame.mm(px))

    placer = _Placer(canvas.W, canvas.H, [title, legend, header_box, footer, triad_box, ruler_box],
                     coverage, pad=canvas.pt(14), gap=canvas.pt(8))
    # Arrows are content too: keep tags off their shafts where possible.
    for tail, head, _ in arrows:
        t2, h2 = frame.px(_project(np.vstack([tail, head]), view)[:, :2])
        mid = 0.5 * (t2 + h2)
        placer.blocked.append((mid[0] - 3, mid[1] - 3, mid[0] + 3, mid[1] + 3))
    centre_px = frame.px(V[surface_nodes][:, :2]).mean(axis=0)
    radii = [canvas.pt(r) for r in (16, 28, 44, 64, 90)]
    for letter, item in lettered:
        point, visible = anchors.get(letter, (None, False))
        if kinds[letter] == "Gravity" and gravity_arrow is not None:
            anchor = frame.px(gravity_arrow[1][:2])
        elif point is not None:
            anchor = frame.px(np.asarray(point, dtype=float)[:2])
        else:
            continue
        text = letter if visible else f"{letter} (behind)"
        size = _tag_size(canvas, text, 9.5)
        direction = anchor - centre_px
        norm = float(np.linalg.norm(direction))
        prefer = direction / norm if norm > 1e-9 else np.array([0.0, 1.0])
        tag_box = placer.place(anchor, size, radii=radii, avoid_model=1.0, prefer=prefer)
        _tag(canvas, tag_box, anchor, text, face=item_colour[letter], ink="#ffffff", size=9.5,
             marker=kinds[letter] != "Gravity" and visible)
    return canvas.save(out_path)


# --------------------------------------------------------------------------
# Face IDs
# --------------------------------------------------------------------------


def _face_adjacency(tris: np.ndarray, owner: np.ndarray) -> dict[int, set[int]]:
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    f = np.tile(owner, 3)
    e = np.sort(e, axis=1)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, f = e[order], f[order]
    same = np.all(e[1:] == e[:-1], axis=1) & (f[1:] != f[:-1])
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in zip(f[:-1][same].tolist(), f[1:][same].tolist()):
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def _colour_faces(tags: Sequence[int], adjacency: Mapping[int, set[int]],
                  areas: Mapping[int, float]) -> dict[int, int]:
    """Greedy graph colouring, most-connected faces first, so neighbours never
    share a colour (tab20 cycled by tag number did, on any part with >20 faces)."""
    order = sorted(tags, key=lambda t: (-len(adjacency.get(t, ())), -areas.get(t, 0.0), t))
    colour: dict[int, int] = {}
    used = np.zeros(len(FACE_PALETTE), dtype=np.int64)
    for tag in order:
        taken = {colour[n] for n in adjacency.get(tag, ()) if n in colour}
        free = [c for c in range(len(FACE_PALETTE)) if c not in taken]
        pick = min(free, key=lambda c: (used[c], c)) if free else int(np.argmin(used))
        colour[tag] = pick
        used[pick] += 1
    return colour


def _view_name(view: tuple[float, float]) -> str:
    eye = _basis(*view)[2]
    parts = []
    for axis, value in zip("XYZ", eye):
        if abs(value) > 0.2:
            parts.append(("+" if value > 0 else "−") + axis)
    return "view from " + " ".join(parts)


def render_face_ids(mesh: Any, out_path: str | Path, *, dpi: int = 150) -> Path:
    """Every CAD face numbered, from two opposite iso views side by side.

    The numbers are the face tags of the face table. A number is drawn only in
    the view where its face is (most) visible, on the face itself, so a person
    can point at it; faces hidden in both views are listed in the footer rather
    than drawn through the part. Opaque, back-face culled and coloured so that
    touching faces never share a colour.
    """
    from matplotlib.colors import to_rgba_array

    out_path = Path(out_path)
    nodes = np.asarray(mesh.nodes, dtype=float)
    tris, parent = _surface(mesh)
    owner = np.asarray(mesh.boundary_face, dtype=np.int64)[parent]
    tags = sorted(int(t) for t in np.unique(owner))
    tri_area = 0.5 * np.linalg.norm(np.cross(nodes[tris[:, 1]] - nodes[tris[:, 0]],
                                             nodes[tris[:, 2]] - nodes[tris[:, 0]]), axis=1)
    areas = {t: float(tri_area[owner == t].sum()) for t in tags}
    colour_of = _colour_faces(tags, _face_adjacency(tris, owner), areas)
    palette = to_rgba_array(FACE_PALETTE)
    lines = _chain(_cad_edges(mesh))

    views = [DEFAULT_VIEW, (-DEFAULT_VIEW[0], DEFAULT_VIEW[1] + 180.0)]
    canvas = _Canvas(13.0, 6.6, dpi, background="#f4f5f7")
    margin = canvas.inch(0.16)
    part = getattr(mesh, "path", None)
    head = canvas.text(margin, canvas.H - margin,
                       f"Face IDs{' — ' + Path(part).name if part else ''}",
                       fontsize=12, fontweight="bold", ha="left", va="top", color="#1f2d3d", zorder=40)
    head_box = canvas.extent(head)

    panels = []
    for view in views:
        V = _project(nodes, view)
        front = _area2(V, tris) > 0
        ftris = tris[front]
        depth = _DepthMap(V, ftris, count=True)
        visible = np.zeros(len(tris), dtype=bool)
        tol = _tri_tolerance(V, ftris, depth.pix)
        visible[np.flatnonzero(front)] = (depth.tri_pixels > 0) | depth.visible(
            V[ftris].mean(axis=1), tol, loose=False)
        pixels = np.zeros(len(tris))
        pixels[np.flatnonzero(front)] = depth.tri_pixels
        seen = {t: float(pixels[owner == t].sum()) + (0.5 if (visible & (owner == t)).any() else 0.0)
                for t in tags}
        poles = depth.poles(owner[front])
        panels.append(dict(view=view, V=V, front=front, ftris=ftris, depth=depth,
                           visible=visible, seen=seen, poles=poles))

    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    # Both panels share one scale, so the part is the same size in each.
    surface_nodes = np.unique(tris)
    bottom = margin + 2 * canvas.measure("Xg", fontsize=8.5)[1] + canvas.pt(16)
    top = head_box[1] - canvas.pt(26)
    half = canvas.W / 2
    targets = [(margin, bottom, half - margin / 2, top),
               (half + margin / 2, bottom, canvas.W - margin, top)]
    k = min(_fit(p["V"][surface_nodes], tgt, margin=0.06).k for p, tgt in zip(panels, targets))
    frames = []
    for p, tgt in zip(panels, targets):
        xy = p["V"][surface_nodes][:, :2]
        centre = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
        frames.append(_Frame(k, 0.5 * (tgt[0] + tgt[2]) - k * centre[0],
                             0.5 * (tgt[1] + tgt[3]) - k * centre[1]))

    hidden: list[int] = []
    assignment: dict[int, int] = {}
    for t in tags:
        scores = [p["seen"][t] for p in panels]
        if max(scores) <= 0:
            hidden.append(t)
            continue
        # The view with the most room around the label spot, so a ring or a
        # strip gets its number where it is widest, not where it is largest.
        room = [p["poles"][t][1] if t in p["poles"] else -1.0 for p in panels]
        assignment[t] = int(np.argmax(room)) if max(room) > 0 else int(np.argmax(scores))

    placers = []
    for p, frame, tgt in zip(panels, frames, targets):
        V, ftris = p["V"], p["ftris"]
        fowner = owner[p["front"]]
        colours = _shade(palette[[colour_of[int(t)] for t in fowner]],
                         _smooth_normals(V, ftris, fowner))
        order = np.argsort(V[ftris][..., 2].mean(axis=1), kind="stable")
        _draw_polys(canvas, frame.px(V[ftris][order][..., :2]), colours[order], zorder=1)
        runs = _visible_runs(lines, V, p["depth"], 2.0 / frame.k, 2.0 * p["depth"].pix)
        _draw_runs(canvas, [frame.px(r) for r in runs], (0.12, 0.12, 0.12, 0.95), 0.7, zorder=3)
        model_top = float(frame.px(V[surface_nodes][:, :2])[:, 1].max())
        caption = canvas.text(0.5 * (tgt[0] + tgt[2]), min(model_top + canvas.pt(18), top),
                              _view_name(p["view"]), fontsize=9.5, ha="center", va="bottom",
                              color="#333333", zorder=40)
        placers.append(_Placer(canvas.W, canvas.H, [head_box, canvas.extent(caption)]))

    # Largest faces first: they have room to keep their number on the face, and
    # small faces then move off to a leader line rather than covering them.
    label_size = 10.0 if len(tags) <= 40 else 8.0
    radii = [0.0] + [canvas.pt(r) for r in (14, 24, 36, 52, 72, 96)]
    for t in sorted(assignment, key=lambda t: -areas[t]):
        i = assignment[t]
        V, frame = panels[i]["V"], frames[i]
        text = str(t)
        w, h = canvas.measure(text, fontsize=label_size, fontweight="bold")
        size = (w + canvas.pt(7), h + canvas.pt(5))
        pole = panels[i]["poles"].get(t)
        if pole is not None:
            # The visible point deepest inside the face: never in a hole and
            # never on a neighbour's rim. A label wider than the room there
            # moves off with a leader instead of covering the neighbours.
            anchor = frame.px(pole[0])
            fits = pole[1] * frame.k >= 0.5 * math.hypot(*size)
            box = placers[i].place(anchor, size, radii=radii if fits else radii[1:])
        else:
            pool = np.flatnonzero((owner == t) & panels[i]["visible"])
            cen = V[tris[pool]].mean(axis=1)
            weights = np.maximum(tri_area[pool], 1e-12)
            centre = (cen * weights[:, None]).sum(axis=0) / weights.sum()
            anchor_tri = pool[int(np.argmin(np.linalg.norm(cen[:, :2] - centre[:2], axis=1)))]
            anchor = frame.px(V[tris[anchor_tri]].mean(axis=0)[:2])
            box = placers[i].place(anchor, size, radii=radii)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if math.hypot(cx - anchor[0], cy - anchor[1]) > max(size) * 0.6:
            ex = min(max(anchor[0], box[0]), box[2])
            ey = min(max(anchor[1], box[1]), box[3])
            for artist in (
                Line2D([ex, anchor[0]], [ey, anchor[1]], color="#333333", linewidth=0.8, zorder=48),
                Line2D([anchor[0]], [anchor[1]], marker="o", markersize=2.8, color="#333333", zorder=49),
            ):
                artist.set_clip_on(False)
                canvas.ax.add_line(artist)
        patch = FancyBboxPatch((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                               boxstyle=f"round,pad=0,rounding_size={canvas.pt(3)}",
                               facecolor="#ffffff", edgecolor="#333333", linewidth=0.7,
                               zorder=50, alpha=0.95)
        patch.set_clip_on(False)
        canvas.ax.add_patch(patch)
        canvas.text(cx, cy, text, fontsize=label_size, fontweight="bold", color="#111111",
                    ha="center", va="center", zorder=51)

    footer = f"{len(tags)} faces · numbers match the face table"
    if hidden:
        footer += f" · not visible in either view: {', '.join(map(str, hidden))}"
    _footer(canvas, [(footer, "#333333")], margin, margin * 0.7, canvas.W - 2 * margin)
    return canvas.save(out_path)


# --------------------------------------------------------------------------
# Board
# --------------------------------------------------------------------------


def compose_board(images: Sequence[str | Path], out_path: str | Path, *,
                  title: str | None = None, dpi: int = 150) -> Path:
    """Put finished pictures side by side on one sheet for sharing.

    Two per row (three when there are five or more), each at its own aspect.
    Only PNG/JPEG inputs, because they are what the renderers write and what
    matplotlib can read without extra codecs.
    """
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    paths = [Path(p) for p in images]
    if not paths:
        raise StructuralRenderError("compose_board needs at least one image path")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise StructuralRenderError(f"cannot compose a board: missing image(s) {', '.join(missing)}")
    bad = [str(p) for p in paths if p.suffix.lower() not in (".png", ".jpg", ".jpeg")]
    if bad:
        raise StructuralRenderError(
            f"compose_board reads PNG or JPEG only; render these as .png first: {', '.join(bad)}"
        )
    pictures = [mpimg.imread(str(p)) for p in paths]
    n = len(pictures)
    cols = 1 if n == 1 else (2 if n <= 4 else 3)
    rows = int(math.ceil(n / cols))
    cell_w = 6.4
    aspects = [pic.shape[0] / pic.shape[1] for pic in pictures]
    row_h = [cell_w * max(aspects[r * cols:(r + 1) * cols]) for r in range(rows)]
    title_h = 0.5 if title else 0.15
    gap = 0.12
    width = cols * cell_w + (cols + 1) * gap
    height = sum(row_h) + (rows + 1) * gap + title_h
    fig = Figure(figsize=(width, height), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#ffffff")
    if title:
        fig.text(gap / width, 1 - 0.12 / height, title, fontsize=13, fontweight="bold",
                 ha="left", va="top", color="#1f2d3d")
    y_top = height - title_h
    for r in range(rows):
        y_top -= gap
        for c in range(cols):
            i = r * cols + c
            if i >= n:
                break
            pic = pictures[i]
            h_in = cell_w * aspects[i]
            x = gap + c * (cell_w + gap)
            y = y_top - h_in
            ax = fig.add_axes((x / width, y / height, cell_w / width, h_in / height))
            ax.imshow(pic, interpolation="lanczos")
            ax.set_axis_off()
        y_top -= row_h[r]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    return out_path


def choose_result_view(solution: Any) -> tuple[float, float]:
    """The camera for both result pictures (see `_result_view`).

    Public because the caller must use it: drawing results with the setup
    camera (`choose_view`) can turn the hotspot away from the reader, leaving
    the Max tag pointing at a blue region on the near side.
    """
    return _result_view(solution)


__all__ = [
    "ANSYS_LEGEND_COLORS",
    "DEFAULT_VIEW",
    "QUANTITIES",
    "StructuralRenderError",
    "choose_result_view",
    "choose_view",
    "compose_board",
    "render_face_ids",
    "render_result",
    "render_setup",
    "structural_header",
]
