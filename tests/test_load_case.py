"""The interview: what it accepts, and what it refuses rather than ignores.

The dangerous failure here is not a crash. It is accepting a question the model
cannot answer and returning a confident interval anyway -- a gravity load the
training set contains none of, two load sets that cannot be superposed in the
screened quantity, or a support face nobody confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fea_surrogate.load_case import (
    GRAVITY_NEGLIGIBLE_FRACTION,
    LoadCaseError,
    parse,
    propose,
)


@dataclass(frozen=True)
class _Face:
    tag: int
    area: float
    normal: tuple[float, float, float]
    planar: bool = True
    centroid: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class _Solid:
    """Stands in for `cad_intake.Solid` so these run without gmsh."""

    faces: tuple[_Face, ...]
    volume: float

    def face(self, tag: int) -> _Face:
        return next(f for f in self.faces if f.tag == tag)


def _solid(volume: float = 184_000.0) -> _Solid:
    return _Solid(
        faces=(
            _Face(1, 800.0, (0.0, 0.0, 1.0)),
            _Face(2, 800.0, (0.0, 0.0, -1.0)),
            _Face(3, 400.0, (1.0, 0.0, 0.0)),
        ),
        volume=volume,
    )


def _minimal(**over):
    document = {
        "fixed": {"faces": [2]},
        "load": {"faces": [1], "magnitude_n": 5000.0, "direction": [0, 0, -1]},
    }
    document.update(over)
    return document


def test_a_complete_load_case_parses() -> None:
    case = parse(_minimal(), _solid())

    assert case.fixed == {"faces": [2]}
    assert case.load.faces == (1,)
    assert case.load.magnitude_n == 5000.0
    assert case.load.direction == (0.0, 0.0, -1.0)
    assert case.poisson_ratio == 0.3


def test_held_faces_are_required_and_never_defaulted() -> None:
    """An unconstrained body has no stress solution at all."""
    with pytest.raises(LoadCaseError, match="rigid-body freedom"):
        parse({"load": {"faces": [1], "magnitude_n": 100.0}}, _solid())


def test_a_load_is_required() -> None:
    with pytest.raises(LoadCaseError, match="needs a load"):
        parse({"fixed": {"faces": [2]}}, _solid())


def test_a_zero_or_negative_load_is_refused() -> None:
    with pytest.raises(LoadCaseError, match="positive force or pressure"):
        parse(_minimal(load={"faces": [1], "magnitude_n": 0.0}), _solid())


def test_two_load_sets_are_refused_with_the_reason() -> None:
    """Fields superpose; a critical-distance maximum does not."""
    document = {
        "fixed": {"faces": [2]},
        "loads": [
            {"faces": [1], "magnitude_n": 5000.0},
            {"faces": [3], "magnitude_n": 2000.0},
        ],
    }
    with pytest.raises(LoadCaseError, match="maximum of a sum"):
        parse(document, _solid())


def test_one_load_set_in_list_form_is_accepted() -> None:
    document = {"fixed": {"faces": [2]}, "loads": [{"faces": [1], "magnitude_n": 10.0}]}

    assert parse(document, _solid()).load.magnitude_n == 10.0


@pytest.mark.parametrize(
    "analysis,needs",
    [("thermal", "temperature"), ("modal", "eigenvalues"),
     ("contact", "contact"), ("nonlinear", "plasticity")],
)
def test_another_analysis_is_refused_saying_what_it_would_need(analysis, needs) -> None:
    with pytest.raises(LoadCaseError, match=needs):
        parse(_minimal(analysis=analysis), _solid())


def test_a_pressure_resolves_to_the_traction_the_sweep_applied() -> None:
    case = parse(
        _minimal(load={"faces": [1], "pressure_mpa": 2.5, "direction": [0, 0, -1]}),
        _solid(),
    )

    assert case.load.area_mm2 == 800.0
    assert case.load.magnitude_n == pytest.approx(2000.0)
    assert any("resolved to 2000 N" in a for a in case.assumptions())


def test_a_pressure_without_geometry_is_refused_not_guessed() -> None:
    with pytest.raises(LoadCaseError, match="inspect the CAD first"):
        parse(_minimal(load={"faces": [1], "pressure_mpa": 2.5}), solid=None)


def test_giving_both_a_force_and_a_pressure_is_refused() -> None:
    with pytest.raises(LoadCaseError, match="not both"):
        parse(
            _minimal(load={"faces": [1], "magnitude_n": 10.0, "pressure_mpa": 1.0}),
            _solid(),
        )


def test_youngs_modulus_is_recorded_and_its_irrelevance_stated() -> None:
    """Dropping it silently would look like a bug; using it would be wrong."""
    case = parse(
        _minimal(material={"poisson_ratio": 0.3, "youngs_modulus_mpa": 210_000.0}),
        _solid(),
    )

    assert case.youngs_modulus_mpa == 210_000.0
    assert any("independent of E exactly" in a for a in case.assumptions())


def test_negligible_self_weight_proceeds_and_says_so() -> None:
    """184 cm^3 of steel is 14.2 N against 5000 N applied: 0.28%."""
    case = parse(
        _minimal(
            material={"poisson_ratio": 0.3, "density_kg_m3": 7850.0},
            gravity={"enabled": True, "g": 9.80665},
        ),
        _solid(volume=184_000.0),
    )

    assert case.self_weight_n == pytest.approx(14.16, rel=0.01)
    assert any("treated as negligible" in a for a in case.assumptions())


def test_significant_self_weight_is_refused_naming_the_load_pattern() -> None:
    """A body load is outside a training set made only of surface tractions."""
    with pytest.raises(LoadCaseError, match="no body load"):
        parse(
            _minimal(
                load={"faces": [1], "magnitude_n": 50.0},
                material={"poisson_ratio": 0.3, "density_kg_m3": 7850.0},
                gravity={"enabled": True},
            ),
            _solid(volume=184_000.0),
        )


def test_the_negligibility_threshold_is_where_it_says_it_is() -> None:
    solid = _solid(volume=184_000.0)
    weight = 7850.0 * 184_000.0 * 1e-9 * 9.80665
    just_over = weight / GRAVITY_NEGLIGIBLE_FRACTION * 0.99

    with pytest.raises(LoadCaseError, match="outside the domain"):
        parse(
            _minimal(
                load={"faces": [1], "magnitude_n": just_over},
                material={"poisson_ratio": 0.3, "density_kg_m3": 7850.0},
                gravity={"enabled": True},
            ),
            solid,
        )


def test_gravity_without_density_is_refused() -> None:
    with pytest.raises(LoadCaseError, match="density_kg_m3"):
        parse(_minimal(gravity={"enabled": True}), _solid())


def test_an_unstated_value_becomes_an_assumption_and_a_stated_one_does_not() -> None:
    proposed = parse(_minimal(load={"faces": [1], "magnitude_n": 10.0}), _solid())
    stated = parse(
        _minimal(
            load={"faces": [1], "magnitude_n": 10.0, "direction": [0, 0, -1]},
            material={"poisson_ratio": 0.31},
        ),
        _solid(),
    )

    assert any("load direction proposed" in a for a in proposed.assumptions())
    assert any("Poisson ratio proposed" in a for a in proposed.assumptions())
    assert not any("proposed" in a for a in stated.assumptions())


def test_an_unknown_key_is_refused_rather_than_ignored() -> None:
    """A misspelled key that is silently dropped screens the wrong problem."""
    with pytest.raises(LoadCaseError, match="unexpected"):
        parse(_minimal(temperature_c=200.0), _solid())

    with pytest.raises(LoadCaseError, match="unexpected"):
        parse(_minimal(load={"faces": [1], "magnitude_N": 10.0}), _solid())


def test_a_zero_direction_is_refused() -> None:
    with pytest.raises(LoadCaseError, match="zero vector"):
        parse(
            _minimal(load={"faces": [1], "magnitude_n": 10.0, "direction": [0, 0, 0]}),
            _solid(),
        )


def test_the_direction_is_normalised() -> None:
    case = parse(
        _minimal(load={"faces": [1], "magnitude_n": 10.0, "direction": [0, 0, -7]}),
        _solid(),
    )

    assert case.load.direction == (0.0, 0.0, -1.0)


def test_a_proposal_carries_its_reasons_and_what_is_still_missing() -> None:
    suggestion = propose(_solid())

    assert suggestion["proposed"]["fixed"]["faces"]
    assert suggestion["why"]["fixed"].startswith("face ")
    assert any("magnitude_n" in q for q in suggestion["still_needed"])
    # The proposal must never invent a load magnitude.
    assert "magnitude_n" not in suggestion["proposed"]["load"]
