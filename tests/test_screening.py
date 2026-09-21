from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from fea_surrogate.screening import (
    SCREENING_SCHEMA,
    Calibration,
    Prediction,
    ScreeningContractError,
    build_document,
    validate_document,
)


DIGEST = "a" * 64


def _calibration(**overrides: Any) -> Calibration:
    values: dict[str, Any] = {
        "method": "split_conformal_log_residual",
        "set_size": 1464,
        "set_sha256": DIGEST,
        "coverage": 0.9,
        "log_half_width": 0.6,
        "delta": 1e-10,
    }
    values.update(overrides)
    return Calibration(**values)


def _document(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "screened_sha256": DIGEST,
        "model_name": "sfem-mesh-gnn",
        "model_version": "0.2.0",
        "model_sha256": DIGEST,
        "calibration": _calibration(),
        "domain": {"material_class": "linear_elastic", "poisson_ratio": 0.4002},
        "predictions": [Prediction("von_mises_peak", 142.0, 198.0, "MPa")],
        "assumptions": ["linear elastic, small displacement"],
        "limitations": ["screening estimate, not a strength certification"],
    }
    arguments.update(overrides)
    return build_document(**arguments)


def test_a_complete_document_is_built_and_accepted() -> None:
    document = _document()

    assert document["schema_version"] == SCREENING_SCHEMA
    assert document["attestation"] == "screening_estimate"
    assert document["predictions"][0]["units"] == "MPa"
    assert "fields" not in document


def test_the_band_is_multiplicative_and_symmetric_in_log_space() -> None:
    """A fixed width would be wrong across five orders of magnitude."""
    calibration = _calibration(log_half_width=math.log(2.0), delta=0.0)

    lo, hi = calibration.band(100.0)

    assert lo == pytest.approx(50.0)
    assert hi == pytest.approx(200.0)


def test_the_band_scales_with_the_prediction() -> None:
    calibration = _calibration(log_half_width=math.log(2.0), delta=0.0)
    small_lo, small_hi = calibration.band(1.0)
    large_lo, large_hi = calibration.band(1000.0)

    assert small_hi / small_lo == pytest.approx(large_hi / large_lo)


def test_a_non_positive_prediction_cannot_be_banded() -> None:
    with pytest.raises(ScreeningContractError, match="non-positive"):
        _calibration(delta=0.0).band(0.0)


def test_an_empty_domain_is_refused_before_it_is_sent() -> None:
    """An empty envelope claims to cover everything."""
    with pytest.raises(ScreeningContractError, match="nonempty object"):
        _document(domain={})


def test_an_empty_prediction_list_is_refused() -> None:
    with pytest.raises(ScreeningContractError, match="nonempty list"):
        _document(predictions=[])


def test_an_inverted_interval_is_refused() -> None:
    with pytest.raises(ScreeningContractError, match="must not exceed"):
        _document(predictions=[Prediction("von_mises_peak", 198.0, 142.0, "MPa")])


@pytest.mark.parametrize("bound", [math.nan, math.inf, -math.inf])
def test_a_non_finite_bound_is_refused(bound: float) -> None:
    with pytest.raises(ScreeningContractError, match="finite"):
        _document(predictions=[Prediction("von_mises_peak", bound, 198.0, "MPa")])


def test_duplicate_quantities_are_refused() -> None:
    p = Prediction("von_mises_peak", 1.0, 2.0, "MPa")
    with pytest.raises(ScreeningContractError, match="unique"):
        _document(predictions=[p, p])


@pytest.mark.parametrize("key", ["assumptions", "limitations"])
def test_assumptions_and_limitations_are_mandatory(key: str) -> None:
    with pytest.raises(ScreeningContractError, match="nonempty list"):
        _document(**{key: []})


@pytest.mark.parametrize("coverage", [0.0, 1.0, 1.5])
def test_coverage_must_lie_strictly_inside_zero_and_one(coverage: float) -> None:
    with pytest.raises(ScreeningContractError, match="strictly between"):
        _document(calibration=_calibration(coverage=coverage))


def test_a_field_without_per_node_intervals_is_refused() -> None:
    """A contour plot with no uncertainty behind it must not be storable."""
    field = {
        "quantity": "von_mises",
        "mesh_sha256": DIGEST,
        "values_relative_path": "screening/v.json",
        "image_relative_path": "screening/v.png",
        "image_sha256": DIGEST,
    }
    with pytest.raises(ScreeningContractError, match="field entry is incomplete"):
        _document(fields=[field])


@pytest.mark.parametrize(
    "path", ["/abs/v.json", "../escape/v.json", "a\\b.json", "C:/v.json"]
)
def test_a_field_path_must_stay_relative(path: str) -> None:
    field = {
        "quantity": "von_mises",
        "mesh_sha256": DIGEST,
        "values_relative_path": path,
        "interval_relative_path": "screening/i.json",
        "image_relative_path": "screening/v.png",
        "image_sha256": DIGEST,
    }
    with pytest.raises(ScreeningContractError, match="relative|safe path"):
        _document(fields=[field])


def test_attestation_cannot_be_upgraded_to_evidence() -> None:
    document = _document()
    document["attestation"] = "host_verified"

    with pytest.raises(ScreeningContractError, match="attestation must be"):
        validate_document(document)


def test_calibration_loads_from_a_training_run_record(tmp_path: Path) -> None:
    record = {
        "coverage": 0.9,
        "calibration": {
            "method": "split_conformal_log_residual",
            "set_size": 1464,
            "set_sha256": DIGEST,
            "log_half_width": 2.98,
            "delta": 1.3e-10,
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    calibration = Calibration.from_file(path)

    assert calibration.set_size == 1464
    assert calibration.coverage == 0.9
    assert calibration.log_half_width == pytest.approx(2.98)


def test_a_named_family_earns_its_own_band() -> None:
    """One shared band covered welded T joints 0.58 of the time at a claimed 0.90."""
    calibration = _calibration(
        log_half_width=math.log(1.91),
        delta=0.0,
        bands={
            "welded_t": {"log_half_width": math.log(2.99), "set_size": 100},
            "pulley": {"log_half_width": math.log(1.46), "set_size": 100},
        },
    )

    weld_lo, weld_hi = calibration.for_family("welded_t").band(100.0)
    pulley_lo, pulley_hi = calibration.for_family("pulley").band(100.0)

    assert weld_hi / weld_lo > pulley_hi / pulley_lo
    assert weld_hi == pytest.approx(299.0, rel=1e-3)
    assert pulley_hi == pytest.approx(146.0, rel=1e-3)


@pytest.mark.parametrize("family", [None, "unseen_family"])
def test_an_unknown_family_gets_the_widest_band_not_the_average(family) -> None:
    """A group that was never measured has not been shown to be covered."""
    calibration = _calibration(
        log_half_width=math.log(1.91),
        delta=0.0,
        bands={
            "welded_t": {"log_half_width": math.log(2.99), "set_size": 100},
            "pulley": {"log_half_width": math.log(1.46), "set_size": 100},
        },
    )

    lo, hi = calibration.for_family(family).band(100.0)

    assert hi == pytest.approx(299.0, rel=1e-3)


def test_a_calibration_without_bands_is_unchanged_by_family() -> None:
    calibration = _calibration(log_half_width=math.log(2.0), delta=0.0)

    assert calibration.for_family("anything") is calibration


def test_per_family_set_size_is_reported_not_the_total() -> None:
    """The band was measured on that family's residuals, so that is the n."""
    calibration = _calibration(
        set_size=533,
        bands={"pulley": {"log_half_width": 0.4, "set_size": 100}},
    )

    assert calibration.for_family("pulley").set_size == 100
