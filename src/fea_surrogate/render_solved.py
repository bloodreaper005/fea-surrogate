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

from .render import FIELD_COLORMAP, _draw_surface, _face_values, mesh_sha256
from .screening import ScreeningContractError, file_sha256


#: How far an element may sit above its neighbourhood average before the plot
#: calls its value mesh-dependent. 2x is generous -- across the families the
#: raw peak runs 1.9 to 3.3 times the critical-distance value.
#: Burned into every solved render, in place of the screening stamp. A real
#: solve is allowed to look solved; stamping ESTIMATE on the true answer would
#: be its own kind of lie.
SOLVED_HEADER = "SOLVED — linear elastic, P1 tetrahedra"

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
) -> SolvedRender:
    """Two panels: the solved field, and where the mesh is speaking for it."""
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
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    field_axis = figure.add_subplot(1, 2, 1, projection="3d")
    field_axis.set_facecolor("#f4f2ee")
    normal = plt.Normalize(vmin=vmin, vmax=vmax)
    _draw_surface(
        field_axis,
        vertices,
        faces,
        colormaps[FIELD_COLORMAP](normal(_face_values(values, faces))),
        none_refused,
    )
    figure.text(0.26, 0.905, "solved field", ha="center", va="top",
                fontsize=10, color="#333333")
    figure.colorbar(
        ScalarMappable(norm=normal, cmap=FIELD_COLORMAP),
        ax=field_axis, shrink=0.42, pad=0.0, aspect=28,
    ).set_label(f"{quantity} [{units}]", fontsize=8)

    # Right: where the mesh, not the part, is setting the value. Computed from
    # the nodal field rather than the element one, because element indices do
    # not map onto surface triangles and the panel has to be defined everywhere
    # the left panel is.
    nodal_index = _singularity_index(values, solved.critical_distance_mpa)
    singular_axis = figure.add_subplot(1, 2, 2, projection="3d")
    singular_axis.set_facecolor("#f4f2ee")
    singular_norm = plt.Normalize(vmin=0.0, vmax=max(SINGULARITY_RATIO, 1.0) * 1.5)
    _draw_surface(
        singular_axis,
        vertices,
        faces,
        colormaps[SINGULARITY_COLORMAP](
            singular_norm(_face_values(nodal_index, faces))
        ),
        none_refused,
    )
    figure.text(0.74, 0.905, "value ÷ converged value", ha="center", va="top",
                fontsize=10, color="#333333")
    figure.colorbar(
        ScalarMappable(norm=singular_norm, cmap=SINGULARITY_COLORMAP),
        ax=singular_axis, shrink=0.42, pad=0.0, aspect=28,
    ).set_label("× critical-distance value", fontsize=8)

    figure.text(
        0.5, 0.965,
        f"{SOLVED_HEADER} — {quantity} [{units}]",
        ha="center", va="top", fontsize=11, fontweight="bold", color="#1f4e79",
    )
    figure.text(
        0.5, 0.035,
        f"{solved.node_count} nodes, {solved.element_count} elements, "
        f"{solved.seconds:.1f} s · use {solved.critical_distance_mpa:.4g} MPa: "
        f"the peak of {solved.peak_von_mises:.4g} MPa sits at a singularity and "
        "rises without bound under refinement",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )
    figure.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.07, wspace=0.0)
    figure.savefig(image_path, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)

    return SolvedRender(
        image=image_path,
        values=values_path,
        image_sha256=file_sha256(image_path),
        mesh_sha256=mesh_sha256(vertices, faces),
    )


__all__ = [
    "SINGULARITY_COLORMAP",
    "SOLVED_HEADER",
    "SINGULARITY_RATIO",
    "SolvedRender",
    "render_solved",
]
