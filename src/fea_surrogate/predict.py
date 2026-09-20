"""Run the surrogate and emit a `SurrogateScreening/v1` document."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import build_graph, mesh_step, nearest_load_class
from .model import load_model
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

    @classmethod
    def load(
        cls, weights: str | Path, calibration_json: str | Path, *, device: str = "cpu"
    ) -> "Surrogate":
        raw = json.loads(Path(calibration_json).read_text(encoding="utf-8"))
        return cls(
            model=load_model(weights, device=device),
            calibration=Calibration.from_file(calibration_json),
            normaliser=Normaliser.from_file(calibration_json),
            weights_sha256=file_sha256(weights),
            domain=dict(raw.get("domain", {})),
            limitations=list(raw.get("limitations", [])),
        )

    def peak_tau(self, data) -> float:  # noqa: ANN001 - torch_geometric Data
        with torch.no_grad():
            tau = self.normaliser.decode(self.model(data).reshape(-1))
        return float(tau.max().item())


def screen_step(
    surrogate: Surrogate,
    step_path: str | Path,
    *,
    force_n: float,
    characteristic_length_mm: float | None = None,
    fixed_vertices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Screen one STEP file and return a document the consumer will accept.

    The prediction is dimensionless, so converting it to stress needs the part's
    real size and load: sigma[MPa] = tau * F[N] / L[mm]^2, where L is the
    bounding-box diagonal. Pass `characteristic_length_mm` when the STEP file's
    units are not millimetres.
    """
    if force_n <= 0 or not np.isfinite(force_n):
        raise ScreeningContractError("force_n must be a positive, finite number")

    mesh = mesh_step(step_path)
    data = build_graph(mesh, fixed_vertices=fixed_vertices, force_n=force_n)
    tau = surrogate.peak_tau(data)
    lo_tau, hi_tau = surrogate.calibration.band(tau)

    length_mm = (
        float(characteristic_length_mm)
        if characteristic_length_mm is not None
        else mesh.original_diagonal
    )
    if length_mm <= 0:
        raise ScreeningContractError("characteristic length must be positive")

    # sigma[MPa] = tau * F[N] / L[mm]^2, because MPa is N/mm^2.
    factor = force_n / (length_mm ** 2)
    snapped_index, snapped_force = nearest_load_class(force_n)

    domain = dict(surrogate.domain)
    domain["characteristic_length_mm"] = {"min": 1.0, "max": 10_000.0}
    domain["load_class_snapped_to_n"] = snapped_force

    assumptions = [
        "linear elastic, small displacement",
        f"load {force_n:g} N snapped to the trained class {snapped_force:g} N",
        "geometry rescaled into the training bounding box [0,3]^3",
        f"characteristic length taken as {length_mm:g} mm (bounding-box diagonal)",
    ]

    return build_document(
        screened_sha256=file_sha256(step_path),
        model_name="sfem-mesh-gnn",
        model_version="0.2.0",
        model_sha256=surrogate.weights_sha256,
        calibration=surrogate.calibration,
        domain=domain,
        predictions=[
            Prediction(
                quantity="von_mises_peak",
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
