"""Run the surrogate and emit a `SurrogateScreening/v1` document."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    fixed_spec, loaded_spec, direction, origin = resolve(
        family, fixed, loaded, load_direction
    )
    mesh = mesh_step(step_path, version=surrogate.version)
    data = build_graph(
        mesh,
        version=surrogate.version,
        fixed_vertices=select(fixed_spec, mesh.vertices, "fixed"),
        loaded_vertices=select(loaded_spec, mesh.vertices, "loaded"),
        load_direction=tuple(direction),
        force_n=force_n,
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
        limitations=surrogate.limitations
        or ["screening estimate, not a strength certification"],
    )


__all__ = ["Normaliser", "Surrogate", "screen_step"]
