from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")
pytest.importorskip("scipy", reason="solving requires the inference extra")

from fea_surrogate.generate.dataset import SweptDataset  # noqa: E402
from fea_surrogate.generate.sweep import run  # noqa: E402


@pytest.fixture(scope="module")
def swept(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("sweep")
    run(out, samples=6, families=["bracket", "welded_t"], workers=2, shard_size=3)
    return out


def test_a_sweep_writes_shards_and_a_manifest(swept: Path) -> None:
    dataset = SweptDataset(swept)

    assert len(dataset) == 6
    assert len(dataset.shards) == 2
    assert set(dataset.manifest["by_family"]) == {"bracket", "welded_t"}


def test_the_manifest_digest_identifies_the_set(swept: Path) -> None:
    """A coverage figure is only meaningful if its set can be identified."""
    digest = SweptDataset(swept).set_sha256

    assert len(digest) == 64
    assert digest == SweptDataset(swept).set_sha256


def test_samples_round_trip_with_local_indices(swept: Path) -> None:
    for sample in SweptDataset(swept):
        assert sample.x.shape[1] == 13
        assert len(sample.y_nodal) == len(sample.x)
        # Shards store tetrahedra offset; a read sample must stand alone.
        assert sample.tetrahedra.min() >= 0
        assert sample.tetrahedra.max() < len(sample.x)
        assert np.isfinite(sample.tau) and sample.tau > 0.0


def test_edges_are_undirected_and_deduplicated(swept: Path) -> None:
    sample = next(iter(SweptDataset(swept)))
    edges = sample.edge_index()

    assert edges.shape[0] == 2
    assert edges.shape[1] % 2 == 0
    forward = set(map(tuple, edges.T[: edges.shape[1] // 2]))
    backward = set(map(tuple, edges.T[edges.shape[1] // 2:]))
    assert forward == {(b, a) for a, b in backward}


def test_a_second_run_resumes_instead_of_redrawing(swept: Path) -> None:
    before = SweptDataset(swept).set_sha256
    run(swept, samples=6, families=["bracket", "welded_t"], workers=2, shard_size=3)

    assert SweptDataset(swept).set_sha256 == before


def test_taus_can_be_read_without_materialising_meshes(swept: Path) -> None:
    dataset = SweptDataset(swept)

    assert dataset.taus().shape == (len(dataset),)
