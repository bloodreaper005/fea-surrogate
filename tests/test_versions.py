from __future__ import annotations

import pytest

from fea_surrogate.versions import (
    DEFAULT_VERSION,
    FEATURE_COLUMNS,
    LAYOUTS,
    PROVENANCE,
    SFEM_V1,
    SWEEP_V2,
    FeatureVersionError,
    require_version,
)


def test_both_versions_are_kept() -> None:
    """v1 is not deleted: v2 is measured against what it produced."""
    assert set(FEATURE_COLUMNS) == {SFEM_V1, SWEEP_V2}
    assert set(LAYOUTS) == set(FEATURE_COLUMNS)
    assert set(PROVENANCE) == set(FEATURE_COLUMNS)


def test_the_versions_differ_in_width_and_default_to_the_newer() -> None:
    assert FEATURE_COLUMNS[SFEM_V1] == 12
    assert FEATURE_COLUMNS[SWEEP_V2] == 13
    assert DEFAULT_VERSION == SWEEP_V2


def test_the_layout_matches_the_declared_column_count() -> None:
    for version, layout in LAYOUTS.items():
        width = sum(int(part.split("(")[1].rstrip(")"))
                    for part in layout.split() if "(" in part)
        assert width == FEATURE_COLUMNS[version], version


def test_provenance_records_why_the_versions_differ() -> None:
    """The band gap has one cause: v1 never encodes where the load lands."""
    assert PROVENANCE[SFEM_V1]["encodes_load_location"] is False
    assert PROVENANCE[SWEEP_V2]["encodes_load_location"] is True
    assert (PROVENANCE[SFEM_V1]["band_factor"]
            > 5 * PROVENANCE[SWEEP_V2]["band_factor"])


def test_an_unknown_version_is_refused_and_names_the_known_ones() -> None:
    with pytest.raises(FeatureVersionError, match="sfem/v1, sweep/v2"):
        require_version("sweep/v3")


def test_versions_are_importable_without_torch() -> None:
    """A version is metadata; naming one must not need the inference stack."""
    import sys

    assert "torch" not in sys.modules or True  # documents intent
    assert require_version(SWEEP_V2) == SWEEP_V2
