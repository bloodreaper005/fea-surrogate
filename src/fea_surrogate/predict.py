"""Run the surrogate and emit a `SurrogateScreening/v1` document."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .features import build_graph, mesh_step, nearest_load_class
from .versions import DEFAULT_VERSION, SFEM_V1


#: Facts about how a model was made. They describe the estimate; they are not
#: conditions a query can meet, so they must never reach the domain gate.
PROVENANCE_KEYS = frozenset({
    "source_dataset",
    "families",
    "load_pattern",
    "dataset",
    "note",
})
from .model import load_model
from .face_selectors import (
    FAMILY_FACE_PRESETS,
    FaceSelectionError,
    face_mask,
    select_faces,
)
from .neighbours import describe
from .selectors import resolve, select
from .screening import (
    Calibration,
    Prediction,
    ScreeningContractError,
    build_document,
    file_sha256,
)


@dataclass(frozen=True)
class Normaliser:
    """Inverse of the training transform: tau = exp(z * std + mean) - delta."""

    log_mean: float
    log_std: float
    delta: float

    @classmethod
    def from_file(cls, path: str | Path) -> "Normaliser":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))["normaliser"]
        return cls(
            log_mean=float(raw["log_mean"]),
            log_std=float(raw["log_std"]),
            delta=float(raw.get("delta", 0.0)),
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(z * (self.log_std + 1e-5) + self.log_mean) - self.delta


@dataclass(frozen=True)
class Surrogate:
    model: Any
    calibration: Calibration
    normaliser: Normaliser
    weights_sha256: str
    domain: dict[str, Any]
    limitations: list[str]
    quantity: str = "von_mises_critical_distance"
    version: str = DEFAULT_VERSION
    model_name: str = "sweep-mesh-gnn"
    model_version: str = "0.3.0"

    @classmethod
    def load(
        cls, weights: str | Path, calibration_json: str | Path, *, device: str = "cpu"
    ) -> "Surrogate":
        raw = json.loads(Path(calibration_json).read_text(encoding="utf-8"))
        # The feature version is declared by the run that produced the weights.
        # Guessing it would pair a model with a feature construction it was
        # never fitted on, which fails silently rather than loudly.
        version = str(raw.get("features", {}).get("version", DEFAULT_VERSION))
        identity = raw.get("model", {})
        return cls(
            version=version,
            model_name=str(identity.get("name", "sweep-mesh-gnn")),
            model_version=str(identity.get("version", "0.3.0")),
            model=load_model(weights, version=version, device=device),
            calibration=Calibration.from_file(calibration_json),
            normaliser=Normaliser.from_file(calibration_json),
            weights_sha256=file_sha256(weights),
            domain=dict(raw.get("domain", {})),
            limitations=list(raw.get("limitations", [])),
            # Named by the run, not hardcoded. A consumer comparing this
            # against a yield strength has to know it is an averaged stress
            # and not a peak, and only the training run knows which it is.
            quantity=str(
                raw.get("target", {}).get("quantity", "von_mises_critical_distance")
            ),
        )

    def peak_tau(self, data) -> float:  # noqa: ANN001 - torch_geometric Data
        with torch.no_grad():
            out = self.model(data).reshape(-1)
            if self.version == SFEM_V1:
                # v1 predicts a field; its peak is the maximum over nodes.
                return float(self.normaliser.decode(out).max().item())
            # v2 predicts the peak itself, in log space.
            return float(torch.exp(
                out * (self.normaliser.log_std + 1e-5) + self.normaliser.log_mean
            ).item())


#: Families whose shipped calibration does not transfer to real CAD, with the
#: reason. A band measured on held-out samples is only meaningful if a screened
#: part produces the same features the sweep did, and for these it does not.
#:
#: spring: `addPipe` left its profile face in the model, so every swept spring
#: carried a doubled node set on its bottom end -- 49 nodes where the top had 25,
#: two faces 0.044 mm apart sharing none. A STEP export drops the orphan, so a
#: screened spring is not the part the model was fitted on. Measured over 100
#: fresh springs the deployed estimate came in at 0.006 of truth with coverage
#: 0.00, while the same model fed sweep-path features scored 1.02. The geometry
#: is fixed in `generate/families/spring.py`; the shipped weights are not, and
#: clearing this entry needs a spring re-sweep and a retrain.
UNTRANSFERABLE_FAMILIES = {
    "spring": (
        "the shipped weights were fitted on spring geometry carrying a "
        "duplicated end face, which real CAD does not have. Measured on 100 "
        "fresh springs the estimate was 0.006 of the solved value at 0.00 "
        "coverage, so it is refused rather than reported. Re-sweep and retrain "
        "the spring family to clear this."
    ),
}


#: A selector naming faces rather than node coordinates. These are the ones that
#: reproduce what the sweep trained on.
_FACE_KEYS = frozenset({"faces", "centroid", "radial"})


def _is_face_selector(spec: Any) -> bool:
    return isinstance(spec, Mapping) and bool(_FACE_KEYS & set(spec))


def _resolve_specs(
    family: str | None,
    fixed: Mapping[str, Any] | None,
    loaded: Mapping[str, Any] | None,
    direction: list[float] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[float], str]:
    """Explicit selectors win; a named family supplies face presets otherwise.

    The presets are the face-based ones. The node-coordinate presets they replace
    selected different regions from the tags the sweep trained on -- a tangent
    sliver of a spring instead of its end coil -- and put deployed coverage at
    0.415 against a calibrated 0.90.
    """
    if fixed is not None and loaded is not None:
        return fixed, loaded, direction or [0.0, 0.0, -1.0], "caller-supplied selectors"
    preset = FAMILY_FACE_PRESETS.get(family or "")
    if preset is None:
        # Fall through to the node-selector resolver, which raises the error
        # explaining that a load case cannot be guessed.
        return resolve(family, fixed, loaded, direction)
    return (
        fixed if fixed is not None else preset["fixed"],
        loaded if loaded is not None else preset["loaded"],
        direction or list(preset["direction"]),
        f"{family!r} family face preset",
    )


def _mask(spec: Mapping[str, Any], mesh, role: str) -> np.ndarray:  # noqa: ANN001
    """A node mask from either a face selector or a node-coordinate selector."""
    if _is_face_selector(spec):
        return face_mask(select_faces(spec, mesh, role), mesh)
    return select(spec, mesh.vertices, role)


def screen_step(
    surrogate: Surrogate,
    step_path: str | Path,
    *,
    force_n: float,
    family: str | None = None,
    fixed: dict[str, Any] | None = None,
    loaded: dict[str, Any] | None = None,
    load_direction: list[float] | None = None,
    characteristic_length_mm: float | None = None,
    neighbours: Any = None,
    poisson_ratio: float = 0.35,
) -> dict[str, Any]:
    """Screen one STEP file and return a document the consumer will accept.

    The prediction is dimensionless, so converting it to stress needs the part's
    real size and load: sigma[MPa] = tau * F[N] / L[mm]^2, where L is the
    bounding-box diagonal. Pass `characteristic_length_mm` when the STEP file's
    units are not millimetres.

    `family` selects the calibration band, and supplies the boundary conditions
    that family was trained under when `fixed` and `loaded` are not given.
    Naming it earns the band that family measured; leaving it out earns the
    widest band any family measured, because a part whose group is unknown has
    not been shown to be covered at the stated rate by anything narrower.

    `fixed` and `loaded` are geometric selectors, for example
    `{"axis": "x", "at": "min"}`. They win over a family preset, because a
    caller who describes the load case means it.
    """
    if force_n <= 0 or not np.isfinite(force_n):
        raise ScreeningContractError("force_n must be a positive, finite number")

    if family in UNTRANSFERABLE_FAMILIES:
        raise ScreeningContractError(
            f"{family}: {UNTRANSFERABLE_FAMILIES[family]}"
        )

    mesh = mesh_step(step_path, version=surrogate.version)
    fixed_spec, loaded_spec, direction, origin = _resolve_specs(
        family, fixed, loaded, load_direction
    )
    fixed_mask = _mask(fixed_spec, mesh, "fixed")
    loaded_mask = _mask(loaded_spec, mesh, "loaded")
    data = build_graph(
        mesh,
        version=surrogate.version,
        fixed_vertices=fixed_mask,
        loaded_vertices=loaded_mask,
        load_direction=tuple(direction),
        force_n=force_n,
        poisson_ratio=poisson_ratio,
    )
    tau = surrogate.peak_tau(data)
    calibration = surrogate.calibration.for_family(family)
    lo_tau, hi_tau = calibration.band(tau)

    length_mm = (
        float(characteristic_length_mm)
        if characteristic_length_mm is not None
        else mesh.original_diagonal
    )
    if length_mm <= 0:
        raise ScreeningContractError("characteristic length must be positive")

    # sigma[MPa] = tau * F[N] / L[mm]^2, because MPa is N/mm^2.
    factor = force_n / (length_mm ** 2)

    # A domain envelope is a set of bounds a query must satisfy. Descriptive
    # metadata is not a bound: gating on `source_dataset` makes every query
    # fail as `missing`, because no caller can answer what a model was trained
    # on. Those facts belong in the assumptions, where they are read rather
    # than matched.
    domain = {
        key: value
        for key, value in surrogate.domain.items()
        if key not in PROVENANCE_KEYS
    }
    provenance = {
        key: value
        for key, value in surrogate.domain.items()
        if key in PROVENANCE_KEYS
    }
    domain["characteristic_length_mm"] = {"min": 1.0, "max": 10_000.0}
    if surrogate.version == SFEM_V1:
        # v1 only. v2 handles force exactly through the dimensionless target,
        # so claiming a snap there would be stating a false assumption.
        domain["load_class_snapped_to_n"] = nearest_load_class(force_n)[1]
    # The band is per family, so the document has to say which one it used.
    # An interval qualified by the wrong group is not qualified.
    domain["calibration_family"] = family or "unknown_widest_band"
    domain["feature_version"] = surrogate.version
    # `origin` is a description of the load case, not a bound a query can
    # satisfy, so it travels in the assumptions. Putting it here would make
    # every query fail as `missing` — the defect this very function fixed for
    # provenance one commit ago.

    assumptions = [
        "linear elastic, small displacement",
        (
            f"load {force_n:g} N snapped to the trained class "
            f"{nearest_load_class(force_n)[1]:g} N"
            if surrogate.version == SFEM_V1
            else f"load {force_n:g} N carried exactly by the dimensionless target"
        ),
        (
            "geometry rescaled into the fixed [0,3]^3 training box"
            if surrogate.version == SFEM_V1
            else "geometry kept in its own millimetres"
        ),
        f"characteristic length taken as {length_mm:g} mm (bounding-box diagonal)",
        (
            f"band from the {family!r} calibration set "
            f"(n={calibration.set_size})"
            if family
            else "no family named, so the widest measured band was used"
        ),
        f"held at {fixed_spec}, loaded at {loaded_spec} "
        f"along {direction}, from {origin}",
        (
            "stress is volume-averaged over one thin dimension, not a peak: "
            "an elastic peak at a re-entrant corner is singular and does not "
            "converge under mesh refinement"
            if "critical_distance" in surrogate.quantity
            else f"quantity is {surrogate.quantity}"
        ),
        *(f"{key}: {value}" for key, value in sorted(provenance.items())),
    ]

    # Retrieval travels in the assumptions and limitations, not in a key of its
    # own: the consumer refuses an unrecognised top-level field, and a neighbour
    # list is context for reading the interval rather than part of the contract.
    extra_limitations: list[str] = []
    if neighbours is not None:
        try:
            found = neighbours.query(
                describe(data.x.numpy(), mesh.tetrahedra), k=5
            )
        except Exception as exc:       # retrieval must never break a screening
            assumptions.append(f"nearest-neighbour retrieval unavailable: {exc}")
        else:
            notes = found.notes(factor)
            if found.extrapolating and len(notes) > 1:
                assumptions.append(notes[0])
                extra_limitations.append(notes[1])
            else:
                assumptions.extend(notes)

    return build_document(
        screened_sha256=file_sha256(step_path),
        model_name=surrogate.model_name,
        model_version=surrogate.model_version,
        model_sha256=surrogate.weights_sha256,
        calibration=calibration,
        domain=domain,
        predictions=[
            Prediction(
                quantity=surrogate.quantity,
                lower=lo_tau * factor,
                upper=hi_tau * factor,
                units="MPa",
            )
        ],
        assumptions=assumptions,
        limitations=(
            list(surrogate.limitations)
            or ["screening estimate, not a strength certification"]
        )
        + extra_limitations,
    )


__all__ = ["Normaliser", "Surrogate", "screen_step"]
