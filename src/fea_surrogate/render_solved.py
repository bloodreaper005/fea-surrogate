"""Draw a solved field, and let it look solved.

`render.py` goes to some trouble to stop a *predicted* field passing for an
analysis: a washed-out palette keyed to confidence, an ESTIMATE watermark, a
header reading NOT AN ANALYSIS. All of that exists because a crisp contour plot
borrows authority its data may not have.

None of it belongs on a real solve. Stamping ESTIMATE across the true answer
would be its own kind of lie, and washing out a field that carries no predictive
uncertainty would hide structure that is actually there. So a solved render gets
its own treatment:

* full saturation, because the field is not an estimate
* a header that says what it is -- solved, and under what idealisation
* a second panel showing what genuinely *is* uncertain, which is not the field
  but the **discretisation**

That second panel matters. A solved field's peak sits at a re-entrant corner and
is singular: refining the mesh raises it without bound, which is the whole reason
the training target is volume-averaged rather than a maximum. So the panel maps
each element's departure from its neighbourhood average, and the regions that
light up are exactly the ones whose value states the mesh rather than the part.
A reader should be able to see which parts of the plot not to trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .render import (
    CONTOUR_LEVELS,
    FIELD_COLORMAP,
    _draw_surface,
    _face_values,
    banded_colours,
    mesh_sha256,
)
from .screening import ScreeningContractError, file_sha256


#: How far an element may sit above its neighbourhood average before the plot
#: calls its value mesh-dependent. 2x is generous -- across the families the
#: raw peak runs 1.9 to 3.3 times the critical-distance value.
#: Burned into every solved render, in place of the screening stamp. A real
#: solve is allowed to look solved; stamping ESTIMATE on the true answer would
#: be its own kind of lie.
SOLVED_HEADER = "SOLVED — linear elastic, P1 tetrahedra"

#: The field is scaled to this percentile rather than its maximum. An elastic
#: peak sits at a singularity and is often several times the stress anywhere
#: else, so scaling to the maximum spends one band on the singular corner and
#: crushes the entire rest of the part into the bottom band. Measured on the
#: L-bracket: max 2753 MPa against a 95th percentile of about 600, so eight of
#: nine bands were unused. Most FEA post-processors clip for the same reason.
FIELD_CLIP_PERCENTILE = 95.0

SINGULARITY_RATIO = 2.0

#: Deliberately not the field's palette, so the two panels cannot be confused.
SINGULARITY_COLORMAP = "OrRd"


@dataclass(frozen=True)
class SolvedRender:
    image: Path
    values: Path
    image_sha256: str
    mesh_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_relative_path": self.image.name,
            "values_relative_path": self.values.name,
            "image_sha256": self.image_sha256,
            "mesh_sha256": self.mesh_sha256,
        }


def _singularity_index(
    element_values: np.ndarray, critical_distance: float
) -> np.ndarray:
    """How far each element sits above the converged value, as a ratio.

    1.0 means the element agrees with the critical-distance average. Much above
    it and the element is reading the singularity rather than the part.
    """
    reference = max(float(critical_distance), np.finfo(float).eps)
    return np.asarray(element_values, dtype=float) / reference


def render_solved(
    solved,  # noqa: ANN001 - solve_cad.SolvedPart
    *,
    out_dir: str | Path,
    quantity: str = "von_mises",
    units: str = "MPa",
    stem: str = "solved",
    dpi: int = 150,
    fmt: str = "png",
    levels: int = CONTOUR_LEVELS,
    clip_percentile: float = FIELD_CLIP_PERCENTILE,
) -> SolvedRender:
    """Two panels: the solved field, and where the mesh is speaking for it.

    `levels` sets the number of discrete contour bands, nine by default -- what
    Ansys and Abaqus show, and for a reason: a continuous ramp says where the
    stress is high, bands let a reader put a number to a region. Pass 0 for a
    continuous ramp instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable

    vertices = np.asarray(solved.vertices, dtype=float)
    faces = np.asarray(solved.surface_faces, dtype=np.int64)
    if faces.ndim == 2 and faces.shape[0] == 3 and faces.shape[1] != 3:
        faces = faces.T
    values = np.asarray(solved.von_mises, dtype=float).reshape(-1)

    if values.shape[0] != vertices.shape[0]:
        raise ScreeningContractError(
            "a solved field must carry one value per vertex"
        )
    if not np.all(np.isfinite(values)):
        raise ScreeningContractError("the solved field is not finite")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values_path = out_dir / f"{stem}_values.npy"
    image_path = out_dir / f"{stem}.{fmt}"
    np.save(values_path, values)

    figure = plt.figure(figsize=(11.0, 4.6))
    figure.patch.set_facecolor("#f4f2ee")
    none_refused = np.zeros(faces.shape[0], dtype=bool)

    # Left: the field, at full saturation. It is not an estimate.
    vmin, vmax = float(values.min()), float(values.max())
    clipped = clip_percentile < 100.0 and values.size > 1
    if clipped:
        top = float(np.percentile(values, clip_percentile))
        # Only clip if it actually buys resolution. On a part with no
        # singularity the percentile sits just under the maximum and clipping
        # would discard range for nothing.
        clipped = top > vmin and vmax > top * 1.2
        if clipped:
            vmax = top
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    field_axis = figure.add_subplot(1, 2, 1, projection="3d")
    field_axis.set_facecolor("#f4f2ee")
    face_field = _face_values(values, faces)
    if levels > 0:
        colours, normal, palette = banded_colours(
            face_field, vmin, vmax, FIELD_COLORMAP, levels
        )
        mappable = ScalarMappable(norm=normal, cmap=palette)
    else:
        normal = plt.Normalize(vmin=vmin, vmax=vmax)
        colours = colormaps[FIELD_COLORMAP](normal(face_field))
        mappable = ScalarMappable(norm=normal, cmap=FIELD_COLORMAP)
    _draw_surface(field_axis, vertices, faces, colours, none_refused)
    figure.text(0.26, 0.905, "solved field", ha="center", va="top",
                fontsize=10, color="#333333")
    bar = figure.colorbar(
        mappable, ax=field_axis, shrink=0.42, pad=0.0, aspect=28,
        # Ticks on the band boundaries, so each band has a number either side
        # of it rather than a colour a reader has to interpolate by eye.
        **({"ticks": normal.boundaries, "spacing": "proportional"}
           if levels > 0 else {}),
    )
    bar.set_label(
        f"{quantity} [{units}]"
        + (f"\ntop band is >= p{clip_percentile:g}" if clipped else ""),
        fontsize=8,
    )
    bar.ax.tick_params(labelsize=7)

    # Right: where the mesh, not the part, is setting the value. Computed from
    # the nodal field rather than the element one, because element indices do
    # not map onto surface triangles and the panel has to be defined everywhere
    # the left panel is.
    nodal_index = _singularity_index(values, solved.critical_distance_mpa)
    singular_axis = figure.add_subplot(1, 2, 2, projection="3d")
    singular_axis.set_facecolor("#f4f2ee")
    top = max(SINGULARITY_RATIO, 1.0) * 1.5
    face_index = np.clip(_face_values(nodal_index, faces), 0.0, top)
    if levels > 0:
        singular_colours, singular_norm, singular_palette = banded_colours(
            face_index, 0.0, top, SINGULARITY_COLORMAP, levels
        )
        singular_mappable = ScalarMappable(
            norm=singular_norm, cmap=singular_palette
        )
    else:
        singular_norm = plt.Normalize(vmin=0.0, vmax=top)
        singular_colours = colormaps[SINGULARITY_COLORMAP](
            singular_norm(face_index)
        )
        singular_mappable = ScalarMappable(
            norm=singular_norm, cmap=SINGULARITY_COLORMAP
        )
    _draw_surface(singular_axis, vertices, faces, singular_colours, none_refused)
    figure.text(0.74, 0.905, "value ÷ converged value", ha="center", va="top",
                fontsize=10, color="#333333")
    singular_bar = figure.colorbar(
        singular_mappable, ax=singular_axis, shrink=0.42, pad=0.0, aspect=28,
        **({"ticks": singular_norm.boundaries, "spacing": "proportional"}
           if levels > 0 else {}),
    )
    singular_bar.set_label("× critical-distance value", fontsize=8)
    singular_bar.ax.tick_params(labelsize=7)

    figure.text(
        0.5, 0.965,
        f"{SOLVED_HEADER} — {quantity} [{units}]",
        ha="center", va="top", fontsize=11, fontweight="bold", color="#1f4e79",
    )
    # Two lines. One was long enough to run off both edges of the image once
    # the clip note was added to it.
    figure.text(
        0.5, 0.045,
        f"{solved.node_count} nodes, {solved.element_count} elements, "
        f"{solved.seconds:.1f} s"
        + (f" · scale clipped at p{clip_percentile:g} ({vmax:.4g} {units})"
           if clipped else ""),
        ha="center", va="bottom", fontsize=8, color="#444444",
    )
    figure.text(
        0.5, 0.008,
        f"use {solved.critical_distance_mpa:.4g} {units}: the peak of "
        f"{solved.peak_von_mises:.4g} {units} sits at a singularity and rises "
        "without bound under refinement",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )
    figure.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.10, wspace=0.0)
    figure.savefig(image_path, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)

    return SolvedRender(
        image=image_path,
        values=values_path,
        image_sha256=file_sha256(image_path),
        mesh_sha256=mesh_sha256(vertices, faces),
    )


__all__ = [
    "FIELD_CLIP_PERCENTILE",
    "SINGULARITY_COLORMAP",
    "SOLVED_HEADER",
    "SINGULARITY_RATIO",
    "SolvedRender",
    "render_solved",
]
