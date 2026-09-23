"""Render a screening estimate as an image, without lending it false authority.

A crisp rainbow contour plot is the most persuasive image in engineering. The
visualization literature calls the effect *cartohypnosis*: sharp isosurfaces
convey an impression of reality that the underlying data does not support. A
predicted field drawn like a solved Ansys plot borrows the authority of a result
it does not have, and images get pasted into review decks long after the JSON
that qualified them has been left behind.

So everything here is built to be legible as an estimate:

* **Saturation is keyed to confidence.** A node whose interval is wide is drawn
  washed out. Certainty is the only thing allowed to be vivid, which inverts the
  usual failure where every region shouts equally loudly.
* **Two coupled panels, never one.** The field and the width of its interval are
  drawn side by side. A reader who sees only the left panel has still been shown
  the right one.
* **Out-of-domain regions are hatched, never interpolated over.** A refusal is a
  result. Smoothing across it would hide the one thing worth seeing.
* **The provenance is burned into the pixels** — coverage, model, version, and
  the word ESTIMATE across the plot — because the pixels travel and the JSON
  does not.
* **The palette is deliberately not a rainbow.** Resembling a solved plot is the
  failure mode, so the resemblance is broken on purpose.

Rendering needs matplotlib, which the contract modules must not require, so it
is imported inside the functions and the `render` extra carries it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .screening import ScreeningContractError, file_sha256


#: Relative half-width at which a region is drawn fully washed out. A band of
#: +/-100% of the local value carries no usable spatial information, so it is
#: shown as carrying none rather than as a slightly paler colour.
WASHOUT_RELATIVE_HALF_WIDTH = 1.0

#: Sequential, perceptually uniform, and pointedly not the rainbow a solver GUI
#: defaults to. Looking like a solved plot is the thing being avoided.
#: Burned into every screening render. Named so a test can assert that a solved
#: render does not carry them -- the two must never be confusable.
SCREENING_HEADER = "SCREENING ESTIMATE — NOT AN ANALYSIS"
SCREENING_WATERMARK = "ESTIMATE"

FIELD_COLORMAP = "cividis"
WIDTH_COLORMAP = "PuBu"


class FieldNotPredicted(RuntimeError):
    """The model emits a scalar, so there is no field to draw.

    Raised rather than handled, because the alternative — spreading one pooled
    number across a mesh and drawing it — is indistinguishable from a real
    prediction at a glance and is exactly the artifact this module exists to
    prevent.
    """


@dataclass(frozen=True)
class FieldArtifacts:
    """What a rendered field leaves on disk, and the entry that cites it."""

    entry: dict[str, Any]
    paths: tuple[Path, ...]


def mesh_sha256(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Digest the geometry a field is defined on.

    The values are meaningless without the mesh they sit on, so the entry binds
    the two together the way the document binds itself to the screened STEP.
    """
    digest = hashlib.sha256()
    for array, label in ((vertices, b"vertices"), (faces, b"faces")):
        contiguous = np.ascontiguousarray(array)
        digest.update(label)
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.astype(np.float64).tobytes()
                      if contiguous.dtype.kind == "f"
                      else contiguous.astype(np.int64).tobytes())
    return digest.hexdigest()


def _confidence(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Map each node's interval width onto [0, 1], where 1 is fully certain.

    Relative rather than absolute, and against a fixed reference rather than the
    field's own range: normalising within the part would make every part look
    equally confident somewhere, which is false for a part that is uncertain
    everywhere.
    """
    scale = np.maximum(np.abs(values), np.finfo(float).eps)
    relative_half = (upper - lower) / (2.0 * scale)
    return np.clip(1.0 - relative_half / WASHOUT_RELATIVE_HALF_WIDTH, 0.0, 1.0)


def _desaturate(rgba: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Pull colour toward neutral where confidence is low.

    Done in HSV so hue survives while any confidence remains: a partly uncertain
    region can still be read for order, it simply cannot be mistaken for a
    measurement. At the washout point hue is gone entirely, because saturation
    is zero and HSV has no hue left to carry. That is the intended message
    rather than a limitation — a band as wide as the value itself supports no
    ordering, and a saturation floor would preserve a ranking the data does not
    have. Out-of-domain regions are hatched instead, so a refusal never has to
    be told apart from a washout by colour alone.
    """
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

    hsv = rgb_to_hsv(rgba[:, :3])
    hsv[:, 1] *= confidence
    # Lift value toward white as well, so a low-confidence patch recedes rather
    # than merely turning grey. Saturation alone still reads as a solid colour.
    hsv[:, 2] = hsv[:, 2] * confidence + (1.0 - confidence) * 0.92
    out = rgba.copy()
    out[:, :3] = hsv_to_rgb(hsv)
    return out


def _face_values(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """One value per triangle, averaged from its three nodes."""
    return values[faces].mean(axis=1)


def _draw_surface(
    axis,  # noqa: ANN001 - matplotlib Axes3D
    vertices: np.ndarray,
    faces: np.ndarray,
    colours: np.ndarray,
    out_of_domain_faces: np.ndarray,
    include: np.ndarray | None = None,
) -> None:
    """Draw the triangles, splitting the hatched ones into their own collection.

    `set_hatch` applies to a whole collection, so an out-of-domain region can
    only be hatched by being a separate collection. That constraint is what
    keeps the refusal visible instead of shaded slightly differently.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    polygons = vertices[faces]
    inside = ~out_of_domain_faces

    if inside.any():
        axis.add_collection3d(
            Poly3DCollection(
                polygons[inside],
                facecolors=colours[inside],
                edgecolors="none",
                linewidths=0.0,
            )
        )
    if out_of_domain_faces.any():
        refused = Poly3DCollection(
            polygons[out_of_domain_faces],
            facecolors=(0.82, 0.82, 0.84, 1.0),
            edgecolors=(0.35, 0.35, 0.38, 1.0),
            linewidths=0.3,
        )
        refused.set_hatch("///")
        axis.add_collection3d(refused)

    # `include` widens the frame for annotations drawn outside the part, such
    # as a load arrow. Framing on the geometry alone silently clips them: the
    # first load-case drawing had an arrow that was never once visible.
    bounds = vertices if include is None else np.vstack([vertices, include])
    lo = bounds.min(axis=0)
    hi = bounds.max(axis=0)
    extent = np.maximum(hi - lo, 1e-9)
    margin = extent * 0.02
    for setter, index in (
        (axis.set_xlim, 0), (axis.set_ylim, 1), (axis.set_zlim, 2)
    ):
        setter(lo[index] - margin[index], hi[index] + margin[index])
    axis.set_axis_off()
    try:
        # Limits per axis plus a box aspect matching the extents keeps the part
        # undistorted while letting it fill the panel. A cube aspect would be
        # equally correct and mostly empty: a 200x40x10 bracket rendered into a
        # cube occupies a twentieth of the frame, which is what the first
        # version of this did.
        axis.set_box_aspect(tuple(extent / extent.max()), zoom=1.25)
    except TypeError:  # pragma: no cover - zoom arrived in matplotlib 3.6
        axis.set_box_aspect(tuple(extent / extent.max()))
    except AttributeError:  # pragma: no cover - very old matplotlib
        pass


def _stamp(figure, coverage: float, model_name: str, model_version: str,
           quantity: str, units: str) -> None:
    """Burn the qualification into the pixels.

    Every claim a reader needs in order to not mistake this for an analysis has
    to survive being screenshotted out of context, so none of it lives in a
    caption or a filename.
    """
    figure.text(
        0.5, 0.965,
        f"{SCREENING_HEADER} — {quantity} [{units}]",
        ha="center", va="top", fontsize=11, fontweight="bold", color="#8a3324",
    )
    figure.text(
        0.5, 0.035,
        f"{coverage:.0%} coverage · {model_name}@{model_version} · "
        f"saturation tracks confidence; hatched regions are out of domain and "
        f"were not estimated",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )
    figure.text(
        0.5, 0.5, SCREENING_WATERMARK,
        ha="center", va="center", fontsize=68, color="#000000",
        alpha=0.06, rotation=30, zorder=10, fontweight="bold",
    )


def render_field(
    *,
    out_dir: str | Path,
    quantity: str,
    units: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage: float,
    model_name: str,
    model_version: str,
    out_of_domain: np.ndarray | None = None,
    stem: str = "field",
    dpi: int = 150,
    fmt: str = "png",
) -> FieldArtifacts:
    """Write a predicted field, its per-node interval, and the two-panel image.

    Returns the `fields[]` entry for `build_document`. The entry cites both the
    values and the interval because the contract refuses one without the other:
    a rendered field with no per-node band is the single artifact this
    subsystem must never produce.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.shape[0] == 3 and faces.ndim == 2 and faces.shape[1] != 3:
        # Accept the (3, F) layout `ScaledMesh` carries as well as (F, 3).
        faces = faces.T
    values, lower, upper = (
        np.asarray(a, dtype=float).reshape(-1) for a in (values, lower, upper)
    )

    n = vertices.shape[0]
    if not (values.shape[0] == lower.shape[0] == upper.shape[0] == n):
        raise ScreeningContractError(
            "field values and interval bounds must carry one entry per vertex"
        )
    if not np.all(np.isfinite(values) & np.isfinite(lower) & np.isfinite(upper)):
        raise ScreeningContractError("field values and bounds must be finite")
    if np.any(lower > upper):
        raise ScreeningContractError("field lower bound exceeds its upper bound")

    node_refused = (
        np.zeros(n, dtype=bool)
        if out_of_domain is None
        else np.asarray(out_of_domain, dtype=bool).reshape(-1)
    )
    if node_refused.shape[0] != n:
        raise ScreeningContractError("out_of_domain must carry one flag per vertex")
    # A triangle touching a refused node is refused. Erring the other way would
    # let the estimate bleed one element into the region it declined to answer.
    face_refused = node_refused[faces].any(axis=1)

    confidence = _confidence(values, lower, upper)
    width = upper - lower

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values_path = out_dir / f"{stem}_values.npy"
    interval_path = out_dir / f"{stem}_interval.npy"
    image_path = out_dir / f"{stem}.{fmt}"
    np.save(values_path, values)
    # Both edges, not the width: a consumer that wants to compare against a
    # yield strength needs the upper bound itself, and cannot recover it from a
    # width without the centre.
    np.save(interval_path, np.stack([lower, upper], axis=1))

    shown = values[~node_refused] if (~node_refused).any() else values
    vmin, vmax = float(np.min(shown)), float(np.max(shown))
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    figure = plt.figure(figsize=(11.0, 4.6))
    figure.patch.set_facecolor("#f4f2ee")

    field_axis = figure.add_subplot(1, 2, 1, projection="3d")
    field_axis.set_facecolor("#f4f2ee")
    normal = plt.Normalize(vmin=vmin, vmax=vmax)
    colours = colormaps[FIELD_COLORMAP](normal(_face_values(values, faces)))
    colours = _desaturate(colours, _face_values(confidence, faces))
    _draw_surface(field_axis, vertices, faces, colours, face_refused)
    # Panel captions live in figure coordinates, not on the axes: a 3D axes
    # scaled up to fill its panel puts its own title behind the header.
    figure.text(0.26, 0.905, "predicted field",
                ha="center", va="top", fontsize=10, color="#333333")
    figure.colorbar(
        ScalarMappable(norm=normal, cmap=FIELD_COLORMAP),
        ax=field_axis, shrink=0.42, pad=0.0, aspect=28,
    ).set_label(f"{quantity} [{units}]", fontsize=8)

    width_axis = figure.add_subplot(1, 2, 2, projection="3d")
    width_axis.set_facecolor("#f4f2ee")
    shown_width = width[~node_refused] if (~node_refused).any() else width
    wmax = float(np.max(shown_width)) if shown_width.size else 1.0
    width_norm = plt.Normalize(vmin=0.0, vmax=wmax or 1.0)
    _draw_surface(
        width_axis, vertices, faces,
        colormaps[WIDTH_COLORMAP](width_norm(_face_values(width, faces))),
        face_refused,
    )
    figure.text(0.74, 0.905, f"interval width at {coverage:.0%} coverage",
                ha="center", va="top", fontsize=10, color="#333333")
    figure.colorbar(
        ScalarMappable(norm=width_norm, cmap=WIDTH_COLORMAP),
        ax=width_axis, shrink=0.42, pad=0.0, aspect=28,
    ).set_label(f"width [{units}]", fontsize=8)

    _stamp(figure, coverage, model_name, model_version, quantity, units)
    figure.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.07, wspace=0.0)
    figure.savefig(image_path, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)

    entry = {
        "quantity": quantity,
        "mesh_sha256": mesh_sha256(vertices, faces),
        "values_relative_path": values_path.name,
        "interval_relative_path": interval_path.name,
        "image_relative_path": image_path.name,
        "image_sha256": file_sha256(image_path),
    }
    return FieldArtifacts(
        entry=entry, paths=(values_path, interval_path, image_path)
    )


def render_load_case(
    *,
    out_dir: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    fixed_mask: np.ndarray,
    loaded_mask: np.ndarray,
    load_direction: Sequence[float],
    origin: str,
    stem: str = "load_case",
    dpi: int = 150,
) -> Path:
    """Draw where the part is held and where it is pushed.

    This is not a prediction and never becomes a `fields[]` entry — it shows the
    inputs, which are known exactly. It earns its place because a wrongly chosen
    boundary condition is the failure this subsystem is least able to detect on
    its own: a bracket held on the wrong face screens cleanly and answers a
    different question than the one asked. Drawing the selection is how a person
    catches that in a second.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.shape[0] == 3 and faces.ndim == 2 and faces.shape[1] != 3:
        faces = faces.T
    fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
    loaded_mask = np.asarray(loaded_mask, dtype=bool).reshape(-1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{stem}.png"

    figure = plt.figure(figsize=(6.0, 5.4))
    figure.patch.set_facecolor("#f4f2ee")
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    axis.set_facecolor("#f4f2ee")
    # The body is translucent because the marks are the point of the drawing.
    # Drawn opaque, matplotlib's 3D compositing hides every held node that
    # happens to face away from the camera, which on an L-bracket is all of
    # them -- the first version of this showed a blank grey part.
    direction = np.asarray(load_direction, dtype=float)
    unit = direction / (float(np.linalg.norm(direction)) or 1.0)
    span = float(np.max(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0
    arrow = None
    if loaded_mask.any():
        # Sized from the part's smallest extent, not its largest: an arrow
        # scaled to the length of a slender bracket dwarfs the bracket.
        thinnest = float(np.min(np.maximum(
            vertices.max(axis=0) - vertices.min(axis=0), 1e-9
        )))
        length = max(thinnest * 2.5, span * 0.12)
        tip = vertices[loaded_mask].mean(axis=0)
        arrow = (tip - unit * length, unit * length * 0.92)

    _draw_surface(
        axis, vertices, faces,
        np.tile((0.62, 0.65, 0.70, 0.22), (faces.shape[0], 1)),
        np.zeros(faces.shape[0], dtype=bool),
        include=None if arrow is None else np.vstack([arrow[0], arrow[0] + arrow[1]]),
    )
    if fixed_mask.any():
        held = vertices[fixed_mask]
        axis.scatter(held[:, 0], held[:, 1], held[:, 2],
                     s=14, c="#1f4e79", depthshade=False, label="held", zorder=5)
    if loaded_mask.any():
        pushed = vertices[loaded_mask]
        axis.scatter(pushed[:, 0], pushed[:, 1], pushed[:, 2],
                     s=14, c="#b3261e", depthshade=False, label="loaded", zorder=5)
        axis.quiver(*arrow[0], *arrow[1], color="#b3261e",
                    linewidth=2.6, arrow_length_ratio=0.3, zorder=6)
    axis.legend(loc="upper right", fontsize=8, frameon=False)
    figure.text(
        0.5, 0.965, "SCREENING LOAD CASE — INPUTS, NOT RESULTS",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#8a3324",
    )
    figure.text(
        0.5, 0.03, f"selection from {origin}",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )
    figure.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
    figure.savefig(image_path, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)
    return image_path


def render_faces(
    solid,  # noqa: ANN001 - cad_intake.Solid
    mesh,   # noqa: ANN001 - features.ScaledMesh
    *,
    out_dir: str | Path,
    stem: str = "faces",
    dpi: int = 150,
    fmt: str = "png",
) -> Path:
    """Number the faces on the part, so a person can point at one.

    The interview asks which face is held. A text table of tags, areas and
    centroids answers that in principle and badly in practice -- nobody holds a
    centroid in their head. This is the picture a pre-processor would give you,
    and it is the difference between the interview being answerable and being
    guessed at.

    Drawn translucent for the same reason `render_load_case` is: matplotlib's 3D
    compositing hides every face pointing away from the camera, and on an
    L-bracket that is most of them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    if not mesh.face_nodes:
        raise ScreeningContractError(
            "this mesh carries no face-to-node map; it cannot be drawn by face"
        )

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim == 2 and faces.shape[0] == 3 and faces.shape[1] != 3:
        faces = faces.T

    # Which CAD face does each surface triangle belong to? By majority of its
    # three nodes, since a triangle on a shared edge touches two faces.
    owner = np.full(vertices.shape[0], -1, dtype=np.int64)
    tags = sorted(mesh.face_nodes)
    # Largest first, so a small face painted after a big one keeps its colour on
    # the shared edge rather than being overwritten by its neighbour.
    for order, tag in enumerate(sorted(tags, key=lambda t: -len(mesh.face_nodes[t]))):
        owner[mesh.face_nodes[tag]] = order
    triangle_owner = np.array(
        [np.bincount(owner[tri][owner[tri] >= 0], minlength=len(tags)).argmax()
         if (owner[tri] >= 0).any() else 0
         for tri in faces],
        dtype=np.int64,
    )
    palette = colormaps["tab20"](np.linspace(0.0, 1.0, max(len(tags), 2)))
    colours = palette[triangle_owner % len(palette)]
    colours[:, 3] = 0.55

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{stem}.{fmt}"

    # Float each label just off its own face, along that face's normal. Placed
    # at the centroid instead, the two faces of a thin plate land on the same
    # screen position and their numbers sit on top of each other -- on an
    # L-bracket, 1 over 8 and 7 over 4, which is worse than no label at all.
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    lift = diagonal * 0.06
    labelled = [f for f in solid.faces if f.tag in mesh.face_nodes]
    anchors = np.array([f.centroid for f in labelled], dtype=float)
    normals = np.array([f.normal for f in labelled], dtype=float)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.where(lengths > 0, lengths, 1.0))
    tips = anchors + normals * lift

    figure = plt.figure(figsize=(7.0, 6.0))
    figure.patch.set_facecolor("#f4f2ee")
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    axis.set_facecolor("#f4f2ee")
    _draw_surface(axis, vertices, faces, colours,
                  np.zeros(faces.shape[0], dtype=bool), include=tips)

    for face, anchor, tip in zip(labelled, anchors, tips):
        # A leader line, so a lifted number still says which face it belongs to.
        axis.plot(*zip(anchor, tip), color="#555555", linewidth=0.7, zorder=11)
        axis.text(*tip, str(face.tag), fontsize=11, fontweight="bold",
                  color="#111111", ha="center", va="center", zorder=12,
                  bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff",
                            edgecolor="#555555", linewidth=0.5, alpha=0.9))

    figure.text(0.5, 0.965, "FACES — name one to hold it or load it",
                ha="center", va="top", fontsize=11, fontweight="bold",
                color="#1f4e79")
    figure.text(0.5, 0.03,
                f"{len(solid.faces)} faces · volume {solid.volume:.4g} · "
                f"numbers match the face table",
                ha="center", va="bottom", fontsize=8, color="#444444")
    figure.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.07)
    figure.savefig(image_path, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)
    return image_path


def field_or_refuse(surrogate, data):  # noqa: ANN001
    """Per-node predictions, or a refusal naming why there are none.

    `sweep/v2` pools the graph and emits one scalar. There is no per-node
    quantity in it to draw, and painting the pooled value across every node
    would produce a uniform plot that looks exactly like a prediction. The
    contract's own rule applies: a point estimate is refused, not widened.
    """
    import torch

    from .versions import SFEM_V1

    if surrogate.version != SFEM_V1:
        raise FieldNotPredicted(
            f"{surrogate.version} has a graph-level head and predicts a single "
            "pooled value; there is no per-node field to render. Training a "
            "nodal head against the sweep's y_nodal labels is what unlocks this."
        )
    with torch.no_grad():
        return surrogate.normaliser.decode(
            surrogate.model(data).reshape(-1)
        ).cpu().numpy()


__all__ = [
    "FIELD_COLORMAP",
    "SCREENING_HEADER",
    "SCREENING_WATERMARK",
    "WASHOUT_RELATIVE_HALF_WIDTH",
    "WIDTH_COLORMAP",
    "FieldArtifacts",
    "FieldNotPredicted",
    "field_or_refuse",
    "mesh_sha256",
    "render_faces",
    "render_field",
    "render_load_case",
]
