from __future__ import annotations

import numpy as np
import pytest

from fea_surrogate.selectors import (
    FAMILY_PRESETS,
    Selector,
    SelectorError,
    resolve,
    select,
)


BOX = np.array([[0, 0, 0], [10, 0, 0], [10, 5, 2], [0, 5, 2], [5, 2.5, 1]], float)


def test_a_planar_extreme_selects_one_end() -> None:
    assert Selector({"axis": "x", "at": "min"}).mask(BOX).sum() == 2
    assert Selector({"axis": "x", "at": "max"}).mask(BOX).sum() == 2


def test_a_band_selects_the_interior() -> None:
    mask = Selector({"axis": "z", "between": [0.2, 0.8]}).mask(BOX)
    assert mask.sum() == 1


def test_a_radial_selector_separates_bore_from_rim() -> None:
    angle = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.concatenate([
        np.stack([2 * np.cos(angle), 2 * np.sin(angle), np.zeros_like(angle)], 1),
        np.stack([9 * np.cos(angle), 9 * np.sin(angle), np.zeros_like(angle)], 1),
    ])
    assert Selector({"radial": "inner"}).mask(ring).sum() == 24
    assert Selector({"radial": "outer"}).mask(ring).sum() == 24


def test_a_selector_matching_nothing_is_refused() -> None:
    """Silently matching nothing misdescribes the load case."""
    with pytest.raises(SelectorError, match="matched no vertex"):
        select({"axis": "x", "between": [0.98, 0.99]}, BOX[:1], "fixed")


@pytest.mark.parametrize("bad", [{"axis": "w", "at": "min"},
                                 {"axis": "x", "at": "middle"},
                                 {"radial": "sideways"},
                                 {"at": "min"}])
def test_an_unreadable_selector_is_refused(bad) -> None:
    with pytest.raises(SelectorError):
        Selector(bad).mask(BOX)


def test_every_family_has_a_preset() -> None:
    """Naming a family reproduces the conditions its band was measured under."""
    from fea_surrogate.generate.families import ALL_FAMILIES

    assert set(FAMILY_PRESETS) == {f.name for f in ALL_FAMILIES}
    for name, preset in FAMILY_PRESETS.items():
        assert set(preset) == {"fixed", "loaded", "direction"}, name


def test_explicit_selectors_beat_a_preset() -> None:
    fixed, loaded, direction, origin = resolve(
        "bracket", {"axis": "y", "at": "max"}, None, None
    )
    assert fixed == {"axis": "y", "at": "max"}
    assert loaded == FAMILY_PRESETS["bracket"]["loaded"]
    assert "caller-supplied" in origin


def test_no_family_and_no_selectors_is_refused() -> None:
    with pytest.raises(SelectorError, match="no boundary conditions"):
        resolve(None, None, None, None)
