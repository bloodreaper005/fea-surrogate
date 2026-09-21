"""The whole chain: STEP file in, SurrogateScreening/v1 document out.

The weights are untrained, so the stress value is meaningless. What is under
test is the plumbing — mesh, features, model shapes, per-family band selection,
unit conversion and the document contract — which is exactly what a real model
cannot tell you is wrong, because a wrong number looks like a number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

gmsh = pytest.importorskip("gmsh", reason="needs the inference extra")
torch = pytest.importorskip("torch", reason="needs the inference extra")
pytest.importorskip("torch_geometric", reason="needs the inference extra")

from fea_surrogate.model import PeakGNN  # noqa: E402
from fea_surrogate.predict import Surrogate, screen_step  # noqa: E402
from fea_surrogate.screening import validate_document  # noqa: E402


@pytest.fixture(scope="module")
def parts(tmp_path_factory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("e2e")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("bracket")
        a = gmsh.model.occ.addBox(0, 0, 0, 180, 60, 12)
        b = gmsh.model.occ.addBox(0, 0, 0, 12, 60, 90)
        gmsh.model.occ.fuse([(3, a)], [(3, b)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(out / "bracket.step"))
    finally:
        gmsh.finalize()

    torch.manual_seed(0)
    torch.save(PeakGNN(13).state_dict(), out / "weights.pt")

    (out / "calibration.json").write_text(json.dumps({
        "model": {"name": "sweep-mesh-gnn", "version": "0.3.0", "sha256": "0" * 64},
        "coverage": 0.9,
        "calibration": {
            "method": "mondrian_split_conformal_log_residual",
            "set_size": 533, "set_sha256": "a" * 64,
            "log_half_width": math.log(1.91), "delta": 0.0,
            "bands": {
                "bracket": {"log_half_width": math.log(1.59), "set_size": 100},
                "welded_t": {"log_half_width": math.log(2.99), "set_size": 100},
            },
        },
        "normaliser": {"log_mean": 6.70, "log_std": 2.86, "delta": 0.0},
        "features": {"version": "sweep/v2", "columns": 13},
        "domain": {"material_class": "linear_elastic",
                   "poisson_ratio": {"min": 0.27, "max": 0.42}},
        "limitations": ["screening estimate, not a strength certification"],
    }), encoding="utf-8")
    return {"step": out / "bracket.step", "weights": out / "weights.pt",
            "calibration": out / "calibration.json"}


@pytest.fixture(scope="module")
def surrogate(parts) -> Surrogate:
    return Surrogate.load(parts["weights"], parts["calibration"])


def test_a_step_file_becomes_a_valid_document(parts, surrogate) -> None:
    document = screen_step(surrogate, parts["step"], force_n=5000.0, family="bracket")

    validate_document(document)
    prediction = document["predictions"][0]
    assert prediction["units"] == "MPa"
    assert prediction["lower"] < prediction["upper"]
    assert document["attestation"] == "screening_estimate"


def test_the_feature_version_comes_from_the_calibration_record(surrogate) -> None:
    """Guessing would pair a model with features it was never fitted on."""
    assert surrogate.version == "sweep/v2"


def test_the_named_family_gets_a_tighter_band_than_an_unnamed_one(
    parts, surrogate
) -> None:
    named = screen_step(surrogate, parts["step"], force_n=5000.0, family="bracket")
    unnamed = screen_step(surrogate, parts["step"], force_n=5000.0)

    def ratio(d):
        p = d["predictions"][0]
        return p["upper"] / p["lower"]

    assert ratio(named) < ratio(unnamed)
    assert named["domain"]["calibration_family"] == "bracket"
    assert unnamed["domain"]["calibration_family"] == "unknown_widest_band"


def test_the_interval_is_linear_in_the_applied_load(parts, surrogate) -> None:
    """sigma = tau F / L^2, so doubling the force doubles the interval."""
    one = screen_step(surrogate, parts["step"], force_n=1000.0, family="bracket")
    two = screen_step(surrogate, parts["step"], force_n=2000.0, family="bracket")

    assert (two["predictions"][0]["upper"]
            == pytest.approx(2 * one["predictions"][0]["upper"], rel=1e-6))


def test_v2_does_not_claim_the_load_was_snapped(parts, surrogate) -> None:
    """Snapping is a v1 artifact; claiming it here would be a false assumption."""
    document = screen_step(surrogate, parts["step"], force_n=5000.0, family="bracket")

    assert not any("snapped" in a for a in document["assumptions"])
    assert "load_class_snapped_to_n" not in document["domain"]
    assert any("dimensionless target" in a for a in document["assumptions"])


def test_a_non_positive_load_is_refused(parts, surrogate) -> None:
    with pytest.raises(Exception, match="positive"):
        screen_step(surrogate, parts["step"], force_n=0.0)


def test_provenance_never_reaches_the_domain_gate(parts, surrogate) -> None:
    """A fact about how a model was made is not a bound a query can satisfy.

    Gating on `source_dataset` made every query fail as `missing`, because no
    caller can answer what a model was trained on.
    """
    document = screen_step(surrogate, parts["step"], force_n=5000.0, family="bracket")

    for descriptive in ("source_dataset", "families", "load_pattern"):
        assert descriptive not in document["domain"]


def test_the_gate_admits_steel_and_refuses_ceramic(parts, surrogate) -> None:
    """Sweeping Poisson's ratio is what lets a steel query through at all."""
    pytest.importorskip("mech_cad_design_agent", reason="consumer not on the path")
    from mech_cad_design_agent.surrogate_domain import evaluate_domain
    from mech_cad_design_agent.surrogate_screening import parse_surrogate_screening

    document = screen_step(surrogate, parts["step"], force_n=5000.0, family="bracket")
    domain = parse_surrogate_screening(document).domain

    def query(nu: float) -> dict:
        return {"material_class": "linear_elastic", "poisson_ratio": nu,
                "characteristic_length_mm": 210.0, "calibration_family": "bracket",
                "feature_version": "sweep/v2"}

    assert evaluate_domain(domain, query(0.30)).in_domain      # steel
    ceramic = evaluate_domain(domain, query(0.22))             # outside the sweep
    assert not ceramic.in_domain
    assert ceramic.violations[0]["key"] == "poisson_ratio"
