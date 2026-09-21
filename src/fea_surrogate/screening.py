"""Build and check `SurrogateScreening/v1` documents.

This is the contract boundary with CAD Agent. The consumer parses these in
`mech_cad_design_agent.surrogate_screening` and refuses anything malformed, so
the rules are mirrored here: a document that would be rejected on arrival is
refused before it is sent, where the error is attributable to this side.

Nothing here imports torch or gmsh. The contract is checkable without a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping


SCREENING_SCHEMA = "SurrogateScreening/v1"
SCREENING_ATTESTATION = "screening_estimate"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ScreeningContractError(ValueError):
    """The document does not satisfy the contract the consumer enforces."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Calibration:
    """What a training run measured about its own error.

    `coverage` is a claim the consumer stores and cannot verify, so it is only
    meaningful alongside `set_size` and `set_sha256`, which say what it was
    measured on.

    Bands are held per family because one shared band is not honest across
    them. Measured on the first swept run, a single band gave 0.90 coverage
    overall while covering welded T joints 0.58 of the time and pulleys 1.00:
    the average was reached by over-covering five families and badly
    under-covering one. `for_family` selects the band that family's own
    residuals earned.
    """

    method: str
    set_size: int
    set_sha256: str
    coverage: float
    log_half_width: float
    delta: float
    bands: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "Calibration":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cal = raw["calibration"]
        return cls(
            method=cal["method"],
            set_size=int(cal["set_size"]),
            set_sha256=cal["set_sha256"],
            coverage=float(raw["coverage"]),
            log_half_width=float(cal["log_half_width"]),
            delta=float(cal.get("delta", 0.0)),
            bands=dict(cal.get("bands", {})),
        )

    def for_family(self, family: str | None) -> "Calibration":
        """The band this family earned, or the widest one if it is unknown.

        An unnamed family gets the worst case rather than the average. A
        screening estimate whose group is unknown has not been shown to be
        covered at the stated rate by anything narrower.
        """
        if family is not None and family in self.bands:
            entry = self.bands[family]
            return replace(
                self,
                log_half_width=float(entry["log_half_width"]),
                set_size=int(entry.get("set_size", self.set_size)),
            )
        if not self.bands:
            return self
        widest = max(
            self.bands.values(), key=lambda e: float(e["log_half_width"])
        )
        return replace(
            self,
            log_half_width=float(widest["log_half_width"]),
            set_size=int(widest.get("set_size", self.set_size)),
        )

    def band(self, value: float) -> tuple[float, float]:
        """Apply the multiplicative band this calibration measured.

        The band is symmetric in log space, which is why it is a factor rather
        than a width: peak stress spans orders of magnitude, and a fixed width
        would be far too wide at the bottom of that range and too tight at the
        top.
        """
        shifted = max(value, 0.0) + self.delta
        if shifted <= 0.0:
            raise ScreeningContractError("cannot band a non-positive prediction")
        centre = math.log(shifted)
        return (
            math.exp(centre - self.log_half_width) - self.delta,
            math.exp(centre + self.log_half_width) - self.delta,
        )


@dataclass(frozen=True)
class Prediction:
    quantity: str
    lower: float
    upper: float
    units: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "lower": self.lower,
            "upper": self.upper,
            "units": self.units,
        }


def build_document(
    *,
    screened_sha256: str,
    model_name: str,
    model_version: str,
    model_sha256: str,
    calibration: Calibration,
    domain: Mapping[str, Any],
    predictions: Iterable[Prediction],
    assumptions: Iterable[str],
    limitations: Iterable[str],
    fields: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Assemble a document and refuse to return one the consumer would reject."""
    document: dict[str, Any] = {
        "schema_version": SCREENING_SCHEMA,
        "screened_sha256": screened_sha256,
        "model": {
            "name": model_name,
            "version": model_version,
            "sha256": model_sha256,
        },
        "coverage": calibration.coverage,
        "calibration": {
            "method": calibration.method,
            "set_size": calibration.set_size,
            "set_sha256": calibration.set_sha256,
        },
        "domain": dict(domain),
        "predictions": [p.as_dict() for p in predictions],
        "assumptions": list(assumptions),
        "limitations": list(limitations),
        "attestation": SCREENING_ATTESTATION,
    }
    field_list = [dict(f) for f in fields]
    if field_list:
        document["fields"] = field_list
    validate_document(document)
    return document


def _require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ScreeningContractError(f"{label} must be a lowercase SHA-256 digest")


def _require_text_list(value: Any, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ScreeningContractError(f"{label} must be a nonempty list")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ScreeningContractError(f"{label} entries must be nonblank strings")


def validate_document(document: Mapping[str, Any]) -> None:
    """Mirror the consumer's parser.

    Kept deliberately strict and deliberately duplicated rather than imported:
    this repository does not depend on CAD Agent, and a contract that is only
    enforced on the receiving side is discovered to be broken in the wrong
    place.
    """
    if document.get("schema_version") != SCREENING_SCHEMA:
        raise ScreeningContractError("schema_version is incompatible")
    if document.get("attestation") != SCREENING_ATTESTATION:
        raise ScreeningContractError(
            f"attestation must be {SCREENING_ATTESTATION!r}"
        )

    _require_digest(document.get("screened_sha256"), "screened_sha256")

    model = document.get("model")
    if not isinstance(model, dict) or set(model) != {"name", "version", "sha256"}:
        raise ScreeningContractError("model identity is incomplete")
    _require_digest(model["sha256"], "model sha256")
    for key in ("name", "version"):
        if not isinstance(model[key], str) or not model[key].strip():
            raise ScreeningContractError(f"model {key} must be a nonblank string")

    coverage = document.get("coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
        raise ScreeningContractError("coverage must be a number")
    if not 0.0 < float(coverage) < 1.0:
        raise ScreeningContractError("coverage must lie strictly between 0 and 1")

    cal = document.get("calibration")
    if not isinstance(cal, dict) or set(cal) != {"method", "set_size", "set_sha256"}:
        raise ScreeningContractError("calibration record is incomplete")
    if (
        isinstance(cal["set_size"], bool)
        or not isinstance(cal["set_size"], int)
        or cal["set_size"] < 1
    ):
        raise ScreeningContractError("calibration set_size must be a positive integer")
    _require_digest(cal["set_sha256"], "calibration set_sha256")

    domain = document.get("domain")
    if not isinstance(domain, dict) or not domain:
        # An empty envelope claims to cover everything, which is the one thing
        # a screening estimate must never claim.
        raise ScreeningContractError("domain envelope must be a nonempty object")

    predictions = document.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ScreeningContractError("predictions must be a nonempty list")
    seen: set[str] = set()
    for item in predictions:
        if not isinstance(item, dict) or set(item) != {
            "quantity",
            "lower",
            "upper",
            "units",
        }:
            raise ScreeningContractError(
                "prediction entry must carry quantity, lower, upper, units"
            )
        quantity = item["quantity"]
        if not isinstance(quantity, str) or not quantity.strip():
            raise ScreeningContractError("prediction quantity must be nonblank")
        if quantity in seen:
            raise ScreeningContractError("prediction quantities must be unique")
        seen.add(quantity)
        for edge in ("lower", "upper"):
            value = item[edge]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ScreeningContractError(f"prediction {edge} must be a number")
            if not math.isfinite(float(value)):
                raise ScreeningContractError(f"prediction {edge} must be finite")
        if float(item["lower"]) > float(item["upper"]):
            raise ScreeningContractError(
                "prediction lower bound must not exceed its upper bound"
            )
        if not isinstance(item["units"], str) or not item["units"].strip():
            raise ScreeningContractError("prediction units must be nonblank")

    _require_text_list(document.get("assumptions"), "assumptions")
    _require_text_list(document.get("limitations"), "limitations")

    if "fields" in document:
        _validate_fields(document["fields"])


def _validate_fields(fields: Any) -> None:
    if not isinstance(fields, list):
        raise ScreeningContractError("fields must be a list")
    required = {
        "quantity",
        "mesh_sha256",
        "values_relative_path",
        "interval_relative_path",
        "image_relative_path",
        "image_sha256",
    }
    for item in fields:
        if not isinstance(item, dict) or set(item) != required:
            # interval_relative_path is required alongside the values: a
            # rendered field with no per-node band is the one artifact this
            # subsystem must not produce.
            raise ScreeningContractError("field entry is incomplete")
        _require_digest(item["mesh_sha256"], "field mesh_sha256")
        _require_digest(item["image_sha256"], "field image_sha256")
        for key in (
            "values_relative_path",
            "interval_relative_path",
            "image_relative_path",
        ):
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                raise ScreeningContractError(f"field {key} must be nonblank")
            if "\\" in value or value.startswith("/") or ":" in value:
                raise ScreeningContractError(f"field {key} must be a relative path")
            if any(part in {"", ".", ".."} for part in value.split("/")):
                raise ScreeningContractError(f"field {key} must be a safe path")


__all__ = [
    "SCREENING_ATTESTATION",
    "SCREENING_SCHEMA",
    "Calibration",
    "Prediction",
    "ScreeningContractError",
    "build_document",
    "file_sha256",
    "validate_document",
]
